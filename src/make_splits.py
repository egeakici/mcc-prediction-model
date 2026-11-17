"""Step 2: immutable, source-independent SHA-1 merchant folds."""
import hashlib
import polars as pl
from .config import (Config, LOG, assert_run, parser, provenance, read_json,
                     read_parquet, sha256, stage, validate_inputs, write_json, write_parquet)


def fold_for_key(key, cfg):
    return int(hashlib.new(cfg.d['split']['hash_algorithm'], key.encode('utf-8')).hexdigest(), 16) % cfg.d['split']['n_folds']


def validate_overlap(cfg, df):
    train = set(df.filter(pl.col('role') == cfg.d['split']['train_role'])['key_loose'])
    val = set(df.filter(pl.col('role') == cfg.d['split']['val_role'])['key_loose'])
    assert not train.intersection(val), 'Train/val merchant overlap'
    with stage(cfg, 'assert merchant separation', len(cfg.d['evaluate']['test_sets']), 'tests') as bar:
        for test in cfg.d['evaluate']['test_sets']:
            keys = set(df.filter(pl.col('role') == test)['key_loose'])
            assert not train.intersection(keys), f'Train/{test} merchant overlap'
            assert not val.intersection(keys), f'Val/{test} merchant overlap'
            LOG.info('%s: train overlap=%s val overlap=%s', test, len(train & keys), len(val & keys))
            bar.update()


def main():
    args = parser(__doc__, force=True).parse_args()
    cfg = Config(args, 'make_splits')
    validate_inputs(cfg)
    assert_run(cfg, read_json(cfg.file('interim', 'build_info')))
    path = cfg.file('splits', 'splits')
    table_hash = sha256(cfg, cfg.file('interim', 'model_table'))
    if path.exists() and not args.force:
        info = read_json(cfg.file('splits', 'split_info'))
        assert_run(cfg, info)
        if info['model_table_sha256'] != table_hash or info['splits_sha256'] != sha256(cfg, path):
            raise ValueError('Existing splits do not match inputs/metadata; use --force to rebuild explicitly')
        validate_overlap(cfg, read_parquet(cfg, path, 'read existing splits'))
        LOG.info('Existing splits kept without overwrite: %s', path)
        return
    df = read_parquet(cfg, cfg.file('interim', 'model_table'), 'read model table')
    keys = sorted(df['key_loose'].unique().to_list())
    with stage(cfg, 'assign SHA-1 folds', len(keys), 'merchants') as bar:
        folds = []
        for key in keys:
            folds.append(fold_for_key(key, cfg))
            bar.update()
    with stage(cfg, 'assign split roles', df.height) as bar:
        df = df.join(pl.DataFrame({'key_loose': keys, 'fold': folds}), on='key_loose', how='left', validate='m:1')
        role = pl.lit(cfg.d['split']['unused_role'])
        for name, spec in cfg.d['split']['roles'].items():
            role = pl.when((pl.col('source_id') == spec['source']) & pl.col('fold').is_in(spec['folds'])).then(pl.lit(name)).otherwise(role)
        anchor = set(df.filter(pl.col('source_id') == cfg.d['split']['anchor_source'])['key_loose'])
        df = df.with_columns(role.alias('role'), pl.col('key_loose').is_in(sorted(anchor)).alias('seen_in_ok'))
        df = df.sort(cfg.d['build']['sort_columns'])
        bar.update(df.height)
    validate_overlap(cfg, df)
    write_parquet(cfg, df, path, 'write immutable splits')
    write_json(cfg, cfg.file('splits', 'split_info'), {**provenance(cfg), 'model_table_sha256': table_hash,
               'splits_sha256': sha256(cfg, path), 'records': df.height})
    LOG.info('Roles: %s', df.group_by('role').agg(pl.len().alias('records'), pl.col('n_rows').sum()).to_dicts())


if __name__ == '__main__':
    main()
