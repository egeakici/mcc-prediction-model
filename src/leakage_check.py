"""Step 5 (optional): random raw-row split as a leakage diagnostic."""
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split
from .build_table import excluded_rows
from .config import Config, LOG, collect, parser, read_parquet, stage, validate_inputs
from .train_tfidf import fit_ovr, load_bundle, score_topk, training_subset, transform, vectorize_fit


def main():
    args = parser(__doc__, rows=True).parse_args()
    cfg = Config(args, 'leakage_check')
    links = validate_inputs(cfg)
    bundle, _ = load_bundle(cfg)
    target = args.rows if args.rows is not None else cfg.d['leakage'][
        'smoke_sample_rows' if cfg.sample is not None else 'sample_rows']
    if target <= 0:
        raise ValueError('--rows must be positive')
    # --sample identifies the upstream smoke run. This diagnostic independently
    # takes the configured 50k random eligible raw rows, never a deduplicated table.
    lf, _, _ = excluded_rows(cfg, cfg.d['leakage']['source'], links)
    raw = collect(cfg, lf.select('text', 'key_loose', 'label_code'), 'read eligible raw rows for random sample')
    n = min(target, raw.height)
    with stage(cfg, 'sample ungrouped raw rows', n) as bar:
        sample = raw.sample(n=n, shuffle=True, seed=cfg.d['seed']).with_columns(pl.lit(1).alias('n_rows'))
        del raw
        bar.update(n)
    with stage(cfg, 'random 80/20 row split', sample.height) as bar:
        train_idx, test_idx = train_test_split(np.arange(sample.height), test_size=cfg.d['leakage']['test_size'],
                                             random_state=cfg.d['seed'], shuffle=True)
        train, test = sample[train_idx], sample[test_idx]
        bar.update(sample.height)
    train, classes, _ = training_subset(cfg, train)
    vecs, x = vectorize_fit(cfg, train['text'].to_list())
    coef, bias, diag = fit_ovr(cfg, x, train['label_code'].to_numpy(), classes, bundle['selected_c'])
    xt = transform(cfg, vecs, test['text'].to_list(), 'transform row-split test')
    top = score_topk(cfg, xt, coef, bias, classes, 'predict row-split test')
    with stage(cfg, 'evaluate row-split leakage', test.height) as bar:
        accuracy = float(np.mean(top[:, 0] == test['label_code'].to_numpy()))
        train_keys = set(train['key_loose'])
        overlap = sum(key in train_keys for key in test['key_loose']) / test.height
        bar.update(test.height)
    results = pl.read_csv(cfg.file('metrics', 'results'))
    comparison = results.filter((pl.col('model') == 'tfidf_linearsvc') &
        (pl.col('test_set') == cfg.d['leakage']['comparison_test']))
    def value(name):
        return comparison.filter(pl.col('metric') == name)['value'][0]
    pl.DataFrame([{'protocol': 'random_raw_row_split', 'acc_rows': accuracy, 'acc_merchant': None,
        'sample_rows': n, 'train_records': train.height, 'test_records': test.height,
        'test_rows': test.height,
        'test_rows_seen_in_train_share': overlap, 'selected_c': bundle['selected_c'],
        'class_count': len(classes), 'convergence_warning_count': len(diag['warnings'])},
        {'protocol': 'merchant_grouped_' + cfg.d['leakage']['comparison_test'], 'acc_rows': value('acc_rows'),
        'acc_merchant': value('acc_merchant'), 'sample_rows': cfg.sample, 'train_records': None,
        'test_records': value('n_text_pairs'), 'test_rows': value('n_rows'), 'test_rows_seen_in_train_share': 0.0,
        'selected_c': bundle['selected_c'], 'class_count': len(bundle['classes']), 'convergence_warning_count': None}]).write_csv(
            cfg.file('metrics', 'leakage'))
    LOG.info('Leakage diagnostic: sampled=%s row accuracy=%s overlap=%s', n, accuracy, overlap)


if __name__ == '__main__':
    main()
