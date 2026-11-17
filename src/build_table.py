"""Step 1: sequential exclusions, untouched no-code rows, deduplicated table."""
import time
import numpy as np
import polars as pl
from .config import (Config, LOG, collect, parser, provenance, sha256, stage,
                     validate_inputs, write_json, write_parquet)
from .text_norm import text_expr


def present(name):
    return pl.col(name).is_not_null() & (pl.col(name).str.strip_chars() != '')


def link_lookup(links):
    lookup = links.select('source_id', 'mcc_desc_norm',
        (pl.col('needs_review').str.to_lowercase() == 'true').alias('_linked_review'))
    if lookup.unique(['source_id', 'mcc_desc_norm']).height != lookup.height:
        raise ValueError('Duplicate description-link keys; cannot safely perform m:1 join')
    return lookup


def excluded_rows(cfg, source, links, *, limit=None, save_no_code=False):
    """Lazy exclusion pipeline, shared unchanged with the row-split diagnostic."""
    lf = pl.scan_parquet(cfg.paths['raw'] / cfg.d['files']['raw'].format(source=source))
    if limit is not None:
        if cfg.d['build']['sample_method'] != 'uniform_without_replacement':
            raise ValueError('Unsupported raw-row sampling method')
        total = collect(cfg, lf.select(pl.len().alias('n')), f'{source}: count sampling population')['n'][0]
        with stage(cfg, f'{source}: choose raw-row sample', min(limit, total)) as bar:
            indices = np.random.default_rng(cfg.d['seed']).choice(total, size=min(limit, total), replace=False)
            bar.update(len(indices))
        sampled = collect(cfg, lf.with_row_index('_sample_row').filter(pl.col('_sample_row').is_in(indices)).drop('_sample_row'),
                          f'{source}: read raw-row sample', len(indices))
        lf = sampled.lazy()
    raw_count = collect(cfg, lf.select(pl.len().alias('n')), f'{source}: count raw')['n'][0]
    rules = [
        ('record_type', (pl.col('record_type') != cfg.d['build']['record_type']).fill_null(True)),
        ('merchant_masked', pl.col('merchant_masked').fill_null(False)),
        ('no_code', ~present('mcc_analysis')),
    ]
    counts = []
    for reason, condition in rules:
        n = collect(cfg, lf.filter(condition).select(pl.len().alias('n')), f'{source}: count {reason}')['n'][0]
        counts.append({'source_id': source, 'reason': reason, 'n_rows': n})
        if reason == 'no_code' and save_no_code:
            with stage(cfg, f'{source}: save no-code rows', n) as bar:
                lf.filter(condition).sink_parquet(cfg.file('interim', 'no_code', source=source),
                    compression=cfg.d['runtime']['parquet_compression'])
                bar.update(n)
        lf = lf.filter(~condition)
    lf = lf.join(link_lookup(links).lazy(), on=['source_id', 'mcc_desc_norm'], how='left', validate='m:1', maintain_order='left')
    linked = ~present('mcc_code')
    missing = collect(cfg, lf.filter(linked & pl.col('_linked_review').is_null()).select(pl.len().alias('n')),
                      f'{source}: validate linked review flags')['n'][0]
    if missing:
        raise ValueError(f'{source}: {missing} linked rows lack a valid description review flag')
    for reason, condition in [
        ('linked_needs_review', linked & pl.col('_linked_review').fill_null(False)),
        ('invalid_key_strict', (~present('key_strict')) |
         (pl.col('key_strict').str.strip_chars().str.len_chars() < cfg.d['build']['min_key_length']))]:
        n = collect(cfg, lf.filter(condition).select(pl.len().alias('n')), f'{source}: count {reason}')['n'][0]
        counts.append({'source_id': source, 'reason': reason, 'n_rows': n})
        lf = lf.filter(~condition)
    lf = lf.with_columns(
        pl.col('mcc_generic').cast(pl.String).str.zfill(cfg.d['build']['code_width']).alias('label_code'),
        pl.col('mcc_group').alias('label_group'),
        pl.when(present('mcc_code')).then(pl.lit('original')).otherwise(pl.lit('linked')).alias('label_origin'),
        text_expr(cfg))
    invalid = collect(cfg, lf.select(
        (~pl.col('label_code').str.contains(cfg.d['build']['code_pattern']).fill_null(False)).sum().alias('bad_code'),
        (~present('key_loose')).sum().alias('bad_key_loose'),
        (~present('label_group')).sum().alias('bad_group'),
        (pl.col('source_id') != source).sum().alias('bad_source')), f'{source}: validate labels and keys')
    if any(invalid.row(0)):
        raise ValueError(f'{source}: invalid data; design decision required: {invalid.to_dicts()}')
    return lf.drop('_linked_review'), counts, raw_count


def main():
    args = parser(__doc__).parse_args()
    cfg = Config(args, 'build_table')
    start = time.perf_counter()
    links = validate_inputs(cfg)
    inputs = [cfg.paths['manifest'], cfg.paths['desc_links'], cfg.paths['taxonomy']] + [
        cfg.paths['raw'] / cfg.d['files']['raw'].format(source=s) for s in cfg.d['sources']]
    with stage(cfg, 'fingerprint inputs', len(inputs), 'files') as bar:
        hashes = {}
        for path in inputs:
            hashes[str(path.relative_to(cfg.root))] = sha256(cfg, path)
            bar.update()
    tables, summaries, exclusions = [], [], []
    for source in cfg.d['sources']:
        lf, counts, raw_count = excluded_rows(cfg, source, links, limit=cfg.sample, save_no_code=True)
        table = collect(cfg, lf.group_by(cfg.d['build']['group_columns']).agg(pl.len().alias('n_rows')),
                        f'{source}: group model table')
        remaining = int(table['n_rows'].sum())
        removed = sum(x['n_rows'] for x in counts)
        assert raw_count == remaining + removed
        summaries.append({'source_id': source, 'raw_rows': raw_count, 'excluded_rows': removed,
            'no_code_rows': next(x['n_rows'] for x in counts if x['reason'] == 'no_code'),
            'remaining_rows': remaining, 'model_records': table.height, 'unique_texts': table['text'].n_unique(),
            'unique_key_loose': table['key_loose'].n_unique(), 'unique_codes': table['label_code'].n_unique()})
        exclusions.extend(counts)
        tables.append(table)
    table = pl.concat(tables).sort(cfg.d['build']['sort_columns'])
    write_parquet(cfg, table, cfg.file('interim', 'model_table'), 'write model table')
    pl.DataFrame(exclusions).write_csv(cfg.file('interim', 'exclusions'))
    pl.DataFrame(summaries).write_csv(cfg.file('interim', 'table_summary'))
    info = {**provenance(cfg), 'input_sha256': hashes, 'elapsed_seconds': time.perf_counter() - start,
            'model_records': table.height, 'represented_rows': int(table['n_rows'].sum()),
            'model_table_sha256': sha256(cfg, cfg.file('interim', 'model_table'))}
    write_json(cfg, cfg.file('interim', 'build_info'), info)
    LOG.info('Build complete: %s', summaries)


if __name__ == '__main__':
    main()
