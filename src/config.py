"""Configuration, guarded output paths, provenance and timed progress."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import platform
import sys
from threading import Event, Thread
import time

import polars as pl
from tqdm.auto import tqdm
import yaml

LOG = logging.getLogger('mcc')


def parser(description, *, force=False, rows=False):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--config', type=Path, default=Path('configs/config.yaml'))
    p.add_argument('--sample', type=int, help='Per-source raw-row limit; isolated smoke outputs')
    if force:
        p.add_argument('--force', action='store_true')
    if rows:
        p.add_argument('--rows', type=int, help='Override leakage sample size only')
    return p


class Config:
    def __init__(self, args, module):
        if args.sample is not None and args.sample <= 0:
            raise ValueError('--sample must be positive')
        self.config_path = args.config.resolve()
        self.root = self.config_path.parent.parent
        self.d = yaml.safe_load(self.config_path.read_text(encoding='utf-8'))
        self.sample = args.sample
        self.module = module
        self.paths = {k: (self.root / v).resolve() for k, v in self.d['paths'].items()}
        if self.sample is not None:
            self.paths.update({k: (self.root / v).resolve() for k, v in self.d['sample_paths'].items()})
        self.protected = [self.paths['raw'], self.paths['reference']]
        for key in self.d['sample_paths']:
            self.guard(self.paths[key])
            self.paths[key].mkdir(parents=True, exist_ok=True)
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, 'reconfigure'):
                stream.reconfigure(encoding=self.d['runtime']['encoding'], errors='replace')
        LOG.setLevel(logging.INFO)
        LOG.handlers.clear()
        handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(
            self.file('logs', 'run_log', module=module), encoding=self.d['runtime']['encoding'])]
        for h in handlers:
            h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
            LOG.addHandler(h)
        self.file('logs', 'config_snapshot', module=module).write_text(
            yaml.safe_dump(self.d, allow_unicode=True, sort_keys=False), encoding='utf-8')
        LOG.info('module=%s sample=%s config_sha256=%s', module, self.sample, self.fingerprint)

    @property
    def fingerprint(self):
        return hashlib.sha256(yaml.safe_dump(self.d, sort_keys=True).encode('utf-8')).hexdigest()

    def guard(self, path):
        path = Path(path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f'Output outside project: {path}')
        if any(path == p or path.is_relative_to(p) for p in self.protected):
            raise ValueError(f'Read-only input path: {path}')
        return path

    def file(self, area, name, **values):
        path = self.guard(self.paths[area] / self.d['files'][name].format(**values))
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def model_file(self, name):
        path = self.guard(self.paths['models'] / self.d['files']['model_dir'] / self.d['files'][name])
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


@contextmanager
def stage(cfg, name, count=None, unit='rows'):
    start = time.perf_counter()
    LOG.info('START %s count=%s unit=%s elapsed=0s', name, count, unit)
    bar = tqdm(total=count, desc=name, unit=unit, mininterval=cfg.d['runtime']['progress_interval'])
    stop = Event()
    def heartbeat():
        while not stop.wait(cfg.d['runtime']['progress_interval']):
            bar.refresh()
    thread = Thread(target=heartbeat, daemon=True)
    thread.start()
    failed = False
    try:
        yield bar
    except BaseException:
        failed = True
        LOG.exception('FAILED %s', name)
        raise
    finally:
        stop.set()
        thread.join()
        bar.close()
        LOG.info('%s %s count=%s unit=%s elapsed=%.3fs', 'FAILED' if failed else 'END',
                 name, bar.n, unit, time.perf_counter() - start)


def collect(cfg, lf, name, count=None):
    with stage(cfg, name, count) as bar:
        df = lf.collect(engine=cfg.d['runtime']['collect_engine'])
        bar.update(int(df['n'][0]) if df.columns == ['n'] and df.height == 1 else df.height)
    return df


def read_parquet(cfg, path, name):
    return collect(cfg, pl.scan_parquet(path), name)


def write_parquet(cfg, df, path, name):
    cfg.guard(path)
    with stage(cfg, name, df.height) as bar:
        df.write_parquet(path, compression=cfg.d['runtime']['parquet_compression'])
        bar.update(df.height)


def write_json(cfg, path, data):
    cfg.guard(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(cfg, path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(cfg.d['runtime']['hash_chunk_size']):
            h.update(chunk)
    return h.hexdigest()


def versions():
    return {'python': platform.python_version(), **{p: importlib.metadata.version(p) for p in
            ['polars', 'pyarrow', 'scikit-learn', 'scipy', 'numpy', 'pyyaml', 'tqdm', 'joblib', 'tabulate']}}


def validate_inputs(cfg):
    # The manifest is always read before scanning any input data.
    text = cfg.paths['manifest'].read_text(encoding='utf-8')
    LOG.info('Manifest read: %s (%s characters)', cfg.paths['manifest'], len(text))
    with stage(cfg, 'validate input schemas', len(cfg.d['sources']), 'sources') as bar:
        for source in cfg.d['sources']:
            path = cfg.paths['raw'] / cfg.d['files']['raw'].format(source=source)
            schema = pl.scan_parquet(path).collect_schema()
            missing = set(cfg.d['required_columns']) - set(schema.names())
            if missing:
                raise ValueError(f'{source}: missing columns {sorted(missing)}')
            bar.update()
    links = pl.read_csv(cfg.paths['desc_links'], infer_schema=False, missing_utf8_is_empty_string=True)
    missing = set(cfg.d['link_columns']) - set(links.columns)
    if missing:
        raise ValueError(f'Link table missing columns: {missing}')
    return links


def assert_run(cfg, info):
    if info['sample'] != cfg.sample or info['config_sha256'] != cfg.fingerprint:
        raise ValueError('Sample size/config differ from upstream artifacts. Rebuild the preceding steps.')


def provenance(cfg):
    return {'sample': cfg.sample, 'config_sha256': cfg.fingerprint, 'versions': versions()}


def taxonomy(cfg):
    with stage(cfg, 'read taxonomy', unit='codes') as bar:
        df = pl.read_csv(cfg.paths['taxonomy'], infer_schema=False, missing_utf8_is_empty_string=True)
        columns = cfg.d['taxonomy_columns']
        df = df.select([pl.col(v).alias(k) for k, v in columns.items()]).unique()
        if df['code'].n_unique() != df.height:
            raise ValueError('Taxonomy contains ambiguous code mappings')
        bar.update(df.height)
    return {r['code']: r for r in df.iter_rows(named=True)}
