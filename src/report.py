"""Step 6: numbers and tables only; no interpretation of results."""
from tabulate import tabulate
import polars as pl
from .config import Config, LOG, parser, read_parquet, stage, taxonomy, validate_inputs
from .train_tfidf import load_bundle


def main():
    args = parser(__doc__).parse_args()
    cfg = Config(args, 'report')
    validate_inputs(cfg)
    _, info = load_bundle(cfg)
    reference = taxonomy(cfg)
    parts = ['# TF-IDF MCC\n']
    def table(title, df):
        def cell(value):
            if isinstance(value, float):
                return format(value, cfg.d['report']['float_format'])
            if isinstance(value, str):
                return value.replace('|', '&#124;').replace('\n', '<br>')
            return value
        parts.append(f'## {title}\n\n' + tabulate([[cell(v) for v in row] for row in df.rows()], headers=df.columns,
            tablefmt=cfg.d['report']['table_format'], floatfmt=cfg.d['report']['float_format'],
            missingval='', disable_numparse=True) + '\n')
    with stage(cfg, 'assemble report tables', unit='tables') as bar:
        table('Veri', pl.read_csv(cfg.file('interim', 'table_summary')))
        table('Dışlamalar (sıralı)', pl.read_csv(cfg.file('interim', 'exclusions')))
        bar.update(2)
        splits = read_parquet(cfg, cfg.file('splits', 'splits'), 'read report splits')
        roles = splits.group_by('role').agg(pl.col('text').n_unique().alias('unique_texts'),
            pl.col('key_loose').n_unique().alias('unique_key_loose'), pl.col('n_rows').sum().alias('rows'),
            pl.col('label_code').n_unique().alias('classes'), pl.len().alias('model_records')).sort('role')
        table('Bölmeler', roles)
        table('Eğitim', pl.DataFrame([{k: info[k] for k in ['sample', 'selected_c', 'train_records',
            'train_unique_texts', 'dropped_train_records', 'dropped_train_rows', 'dropped_train_codes',
            'class_count', 'feature_count', 'elapsed_seconds']}]))
        bar.update(2)
        results = pl.read_csv(cfg.file('metrics', 'results'))
        index = {(r['model'], r['test_set'], r['metric']): r['value'] for r in results.iter_rows(named=True)}
        main_rows, error_rows, origin_rows, seen_rows = [], [], [], []
        def val(model, test, metric):
            return index.get((model, test, metric))
        def with_ci(model, test, metric):
            value, low, high = [val(model, test, metric + suffix) for suffix in ['', '_ci_low', '_ci_high']]
            fmt = cfg.d['report']['float_format']
            if value is None:
                return ''
            if low is None or high is None:
                return format(value, fmt)
            return f'{value:{fmt}} [{low:{fmt}}, {high:{fmt}}]'
        for test in cfg.d['evaluate']['test_sets']:
            for model in cfg.d['train']['model_names']:
                confidence = format(cfg.d['evaluate']['confidence_level'] * 100, 'g')
                row = {'test_set': test, 'model': model, f'acc_rows [{confidence}% GA]': with_ci(model, test, 'acc_rows'),
                    'acc_merchant': val(model, test, 'acc_merchant'), f'macro_f1 [{confidence}% GA]': with_ci(model, test, 'macro_f1')}
                for metric in ['macro_f1_n_classes', 'top3_rows', 'group_acc_rows', 'oov_rate_rows', 'ceiling_rows', 'acc_over_ceiling']:
                    row[metric] = val(model, test, metric)
                main_rows.append(row)
                error_rows.append({'test_set': test, 'model': model, **{m: val(model, test, m) for m in [
                    'in_vocab_error_rows', 'same_group_error_share', 'diff_group_error_share']}})
                for origin in cfg.d['evaluate']['origins']:
                    origin_rows.append({'test_set': test, 'model': model, 'label_origin': origin,
                        'n_rows': val(model, test, f'n_rows_origin_{origin}'), 'acc_rows': val(model, test, f'acc_rows_origin_{origin}')})
                if test in cfg.d['split']['cross_source_tests']:
                    for seen in cfg.d['evaluate']['seen_values']:
                        suffix = str(seen).lower()
                        seen_rows.append({'test_set': test, 'model': model, 'seen_in_ok': seen,
                            'n_rows': val(model, test, f'n_rows_seen_in_ok_{suffix}'),
                            'acc_rows': val(model, test, f'acc_rows_seen_in_ok_{suffix}')})
        table('Ana sonuçlar', pl.DataFrame(main_rows))
        table('Hata ayrıştırması', pl.DataFrame(error_rows))
        table('Etiket kökeni', pl.DataFrame(origin_rows))
        table('Oklahoma kapsamı', pl.DataFrame(seen_rows))
        bar.update(4)
        if cfg.file('metrics', 'leakage').exists():
            table('Sızıntı kontrolü', pl.read_csv(cfg.file('metrics', 'leakage')))
            bar.update()
        table('C araması', pl.read_csv(cfg.file('metrics', 'c_search')))
        bar.update()
        for test in cfg.d['evaluate']['test_sets']:
            for model in cfg.d['train']['model_names']:
                df = read_parquet(cfg, cfg.file('predictions', 'predictions', model=model, test=test),
                                  f'{model}/{test}: read errors')
                errors = (df.filter(pl.col('label_code') != pl.col('pred_code'))
                    .group_by('text', 'label_code', 'pred_code', 'label_group', 'pred_group')
                    .agg(pl.col('n_rows').sum()).sort(['n_rows', 'text', 'label_code', 'pred_code'],
                        descending=[True, False, False, False]).head(cfg.d['report']['top_errors'])
                    .with_columns(
                        pl.col('label_code').replace_strict({k: v['description'] for k, v in reference.items()}, default='', return_dtype=pl.String).alias('true_description'),
                        pl.col('pred_code').replace_strict({k: v['description'] for k, v in reference.items()}, default='', return_dtype=pl.String).alias('pred_description'),
                        (pl.col('label_group') == pl.col('pred_group')).fill_null(False).alias('same_group'))
                    .select('text', 'label_code', 'true_description', 'pred_code', 'pred_description', 'n_rows', 'same_group'))
                table(f'En büyük hatalar: {test} / {model}', errors)
                bar.update()
        path = cfg.file('reports', 'report')
        path.write_text('\n'.join(parts), encoding=cfg.d['runtime']['encoding'])
    LOG.info('Report created: %s tables=%s', path, len(parts) - 1)


if __name__ == '__main__':
    main()
