"""Step 4: weighted rows, unique text pairs and clustered merchant metrics."""
import numpy as np
import polars as pl
from sklearn.metrics import precision_recall_fscore_support
from .config import (Config, LOG, parser, read_parquet, stage, taxonomy, validate_inputs, write_parquet)
from .train_tfidf import load_bundle, score_topk, transform


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def merchant_table(df):
    """Lexical tie breaks after aggregating all records for a label or text."""
    truth = (df.group_by('key_loose', 'label_code').agg(pl.col('n_rows').sum().alias('label_rows'))
             .sort(['key_loose', 'label_rows', 'label_code'], descending=[False, True, False])
             .unique('key_loose', keep='first', maintain_order=True))
    chosen = (df.group_by('key_loose', 'text', 'pred_code').agg(pl.col('n_rows').sum().alias('text_rows'))
              .sort(['key_loose', 'text_rows', 'text', 'pred_code'], descending=[False, True, False, False])
              .unique('key_loose', keep='first', maintain_order=True))
    cluster = df.group_by('key_loose').agg(
        pl.col('n_rows').sum().alias('total_rows'),
        pl.when(pl.col('label_code') == pl.col('pred_code')).then(pl.col('n_rows')).otherwise(0).sum().alias('correct_rows'))
    return truth.join(chosen.select('key_loose', 'text', 'pred_code', 'text_rows'), on='key_loose', validate='1:1').join(
        cluster, on='key_loose', validate='1:1').sort('key_loose')


def f1_from_counts(y, p, eligible, multiplicities, n_codes):
    actual = np.bincount(y, weights=multiplicities, minlength=n_codes)
    predicted = np.bincount(p, weights=multiplicities, minlength=n_codes)
    true_positive = np.bincount(y[y == p], weights=multiplicities[y == p], minlength=n_codes)
    denom = actual + predicted
    scores = np.divide(2 * true_positive, denom, out=np.zeros(n_codes), where=denom != 0)
    return float(scores[eligible].mean()) if len(eligible) else None


def merchant_arrays(cfg, merchants):
    codes = sorted(set(merchants['label_code']) | set(merchants['pred_code']))
    index = {code: i for i, code in enumerate(codes)}
    y = np.asarray([index[c] for c in merchants['label_code']], dtype=np.int64)
    p = np.asarray([index[c] for c in merchants['pred_code']], dtype=np.int64)
    support = np.bincount(y, minlength=len(codes))
    eligible = np.flatnonzero(support >= cfg.d['evaluate']['min_merchants_per_class'])
    return codes, y, p, eligible


def bootstrap(cfg, merchants, arrays):
    codes, y, p, eligible = arrays
    n = merchants.height
    if not n:
        return {f'{metric}_ci_{side}': None for metric in ['acc_rows', 'macro_f1'] for side in ['low', 'high']}
    weights = merchants['total_rows'].to_numpy().astype(np.float64)
    successes = merchants['correct_rows'].to_numpy().astype(np.float64)
    rng = np.random.default_rng(cfg.d['seed'])
    count = cfg.d['evaluate']['bootstrap_replicates']
    accs, f1s = [], []
    with stage(cfg, 'key_loose cluster bootstrap', count, 'replicates') as bar:
        for _ in range(count):
            mult = np.bincount(rng.integers(0, n, size=n), minlength=n)
            accs.append(float(np.dot(mult, successes) / np.dot(mult, weights)))
            f1 = f1_from_counts(y, p, eligible, mult, len(codes))
            if f1 is not None:
                f1s.append(f1)
            bar.update()
    alpha = (1 - cfg.d['evaluate']['confidence_level']) / 2
    result = {}
    for metric, values in [('acc_rows', accs), ('macro_f1', f1s)]:
        ci = np.quantile(values, [alpha, 1 - alpha], method=cfg.d['evaluate']['bootstrap_percentile_method']) if values else [None, None]
        result[f'{metric}_ci_low'], result[f'{metric}_ci_high'] = ci
    return result


def metrics(cfg, df, classes, reference, *, cross_source):
    with stage(cfg, 'aggregate evaluation', df.height) as bar:
        total = int(df['n_rows'].sum())
        correct = pl.col('pred_code') == pl.col('label_code')
        group_correct = (pl.col('pred_group') == pl.col('label_group')).fill_null(False)
        in_vocab = pl.col('label_code').is_in(list(classes))
        wrong = ~correct & in_vocab
        def weight(condition):
            return int(df.filter(condition)['n_rows'].sum())
        pairs = df.unique(['text', 'label_code'])
        merchants = merchant_table(df)
        arrays = merchant_arrays(cfg, merchants)
        codes, y, p, eligible = arrays
        ceiling = ratio(int(merchants['label_rows'].sum()), total)
        accuracy = ratio(weight(correct), total)
        errors = weight(wrong)
        result = {
            'acc_rows': accuracy,
            'acc_texts': ratio(pairs.filter(correct).height, pairs.height),
            'acc_merchant': ratio(int((y == p).sum()), merchants.height),
            'macro_f1': f1_from_counts(y, p, eligible, np.ones(len(y)), len(codes)),
            'macro_f1_n_classes': len(eligible),
            'top3_rows': ratio(weight(pl.col('top3_codes').list.contains(pl.col('label_code'))), total),
            'group_acc_rows': ratio(weight(group_correct), total),
            'oov_rate_rows': ratio(weight(~in_vocab), total),
            'ceiling_rows': ceiling,
            'acc_over_ceiling': ratio(accuracy, ceiling) if accuracy is not None else None,
            'same_group_error_share': ratio(weight(wrong & group_correct), errors),
            'diff_group_error_share': ratio(weight(wrong & ~group_correct), errors),
            'in_vocab_error_rows': errors,
            'n_rows': total, 'n_text_pairs': pairs.height, 'n_merchants': merchants.height,
        }
        for origin in cfg.d['evaluate']['origins']:
            condition = pl.col('label_origin') == origin
            result[f'acc_rows_origin_{origin}'] = ratio(weight(condition & correct), weight(condition))
            result[f'n_rows_origin_{origin}'] = weight(condition)
        if cross_source:
            for seen in cfg.d['evaluate']['seen_values']:
                condition = pl.col('seen_in_ok') == seen
                suffix = str(seen).lower()
                result[f'acc_rows_seen_in_ok_{suffix}'] = ratio(weight(condition & correct), weight(condition))
                result[f'n_rows_seen_in_ok_{suffix}'] = weight(condition)
        if codes:
            precision, recall, f1, support = precision_recall_fscore_support(
                merchants['label_code'].to_numpy(), merchants['pred_code'].to_numpy(), labels=codes,
                zero_division=cfg.d['evaluate']['zero_division'])
            per_class = pl.DataFrame({'code': codes, 'description': [reference.get(c, {}).get('description', '') for c in codes],
                'support': support, 'precision': precision, 'recall': recall, 'f1': f1,
                'macro_f1_eligible': [i in eligible for i in range(len(codes))]})
        else:
            per_class = pl.DataFrame(schema={'code': pl.String, 'description': pl.String, 'support': pl.Int64,
                'precision': pl.Float64, 'recall': pl.Float64, 'f1': pl.Float64, 'macro_f1_eligible': pl.Boolean})
        bar.update(df.height)
    result.update(bootstrap(cfg, merchants, arrays))
    return result, per_class


def predict(cfg, bundle, df, model, reference, x=None):
    if model == 'tfidf_linearsvc':
        top = score_topk(cfg, x, bundle['coef'], bundle['intercept'], bundle['classes'], f'{model}: predict')
    elif model == 'majority':
        with stage(cfg, 'majority: predict', df.height) as bar:
            top = np.tile(np.asarray(bundle['majority_order'][:cfg.d['evaluate']['top_k']]), (df.height, 1))
            bar.update(df.height)
    else:
        raise ValueError(f'Unknown model: {model}')
    predictions = df.with_columns(pl.Series('pred_code', top[:, 0].tolist(), dtype=pl.String),
        pl.Series('top3_codes', top.tolist(), dtype=pl.List(pl.String)))
    return predictions.with_columns(pl.col('pred_code').replace_strict(
        {k: v['group'] for k, v in reference.items()}, default=None, return_dtype=pl.String).alias('pred_group'))


def main():
    args = parser(__doc__).parse_args()
    cfg = Config(args, 'evaluate')
    validate_inputs(cfg)
    bundle, _ = load_bundle(cfg)
    reference = taxonomy(cfg)
    missing = sorted(set(bundle['classes']) - set(reference))
    if missing:
        raise ValueError(f'Training codes missing from taxonomy: {missing}')
    splits = read_parquet(cfg, cfg.file('splits', 'splits'), 'read splits')
    results = []
    for test in cfg.d['evaluate']['test_sets']:
        df = splits.filter(pl.col('role') == test)
        x = transform(cfg, bundle['vectorizers'], df['text'].to_list(), f'{test}: transform') if df.height else None
        for model in cfg.d['train']['model_names']:
            if df.height:
                predictions = predict(cfg, bundle, df, model, reference, x)
            else:
                LOG.warning('%s is empty; undefined metrics exported as blank', test)
                predictions = df.with_columns(pl.lit(None, dtype=pl.String).alias('pred_code'),
                    pl.lit(None, dtype=pl.List(pl.String)).alias('top3_codes'), pl.lit(None, dtype=pl.String).alias('pred_group'))
            columns = ['text', 'key_loose', 'label_code', 'label_group', 'label_origin', 'n_rows',
                       'pred_code', 'top3_codes', 'pred_group', 'seen_in_ok']
            write_parquet(cfg, predictions.select(columns), cfg.file('predictions', 'predictions', model=model, test=test),
                          f'{model}/{test}: save predictions')
            values, per_class = metrics(cfg, predictions, bundle['classes'], reference,
                                        cross_source=test in cfg.d['split']['cross_source_tests'])
            results.extend({'model': model, 'test_set': test, 'metric': key, 'value': value} for key, value in values.items())
            per_class.write_csv(cfg.file('metrics', 'per_class', model=model, test=test))
            LOG.info('%s/%s acc_rows=%s macro_f1=%s eligible_classes=%s', model, test, values['acc_rows'],
                     values['macro_f1'], values['macro_f1_n_classes'])
    pl.DataFrame(results, schema={'model': pl.String, 'test_set': pl.String, 'metric': pl.String, 'value': pl.Float64}).write_csv(
        cfg.file('metrics', 'results'))


if __name__ == '__main__':
    main()
