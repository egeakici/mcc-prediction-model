"""Step 3: train-only TF-IDF and explicit parallel binary OVR LinearSVC."""
import time
import warnings
import joblib
from joblib import Parallel, delayed, parallel_config
import numpy as np
import polars as pl
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from .config import (Config, LOG, assert_run, parser, provenance, read_json, read_parquet,
                     sha256, stage, validate_inputs, write_json)
from .make_splits import validate_overlap


def training_subset(cfg, df):
    counts = df.group_by('label_code').agg(pl.col('text').n_unique().alias('n_texts'))
    counts = counts.filter(pl.col('n_texts') >= cfg.d['train']['min_texts_per_class']).sort('label_code')
    classes = counts['label_code'].to_list()
    selected = df.filter(pl.col('label_code').is_in(classes))
    LOG.info('Train filtering: before=%s after=%s dropped_records=%s dropped_rows=%s dropped_codes=%s',
             df.height, selected.height, df.height - selected.height,
             int(df['n_rows'].sum() - selected['n_rows'].sum()), df['label_code'].n_unique() - len(classes))
    if len(classes) < 2:
        raise ValueError('At least two eligible training classes required; thresholds will not be changed automatically')
    return selected, np.asarray(classes), counts


def vectorize_fit(cfg, texts):
    vectorizers, matrices = [], []
    with stage(cfg, 'fit TF-IDF on train only', len(cfg.d['train']['vectorizers']), 'vectorizers') as bar:
        for name, settings in cfg.d['train']['vectorizers'].items():
            params = dict(settings)
            params['ngram_range'] = tuple(params['ngram_range'])
            params['dtype'] = np.dtype(params['dtype']).type
            vec = TfidfVectorizer(**params)
            matrices.append(vec.fit_transform(texts))
            vectorizers.append((name, vec))
            LOG.info('%s fitted: samples=%s features=%s', name, len(texts), matrices[-1].shape[1])
            bar.update()
    return vectorizers, sparse.hstack(matrices, format='csr')


def transform(cfg, vectorizers, texts, name):
    with stage(cfg, name, len(texts)) as bar:
        result = sparse.hstack([vec.transform(texts) for _, vec in vectorizers], format='csr')
        bar.update(len(texts))
    return result


def _fit_binary(x, y, code, c, settings, seed):
    model = LinearSVC(C=c, random_state=seed, **settings)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(x, (y == code).astype(np.int8))
    messages = [str(w.message) for w in caught]
    return model.coef_[0], model.intercept_[0], int(np.max(model.n_iter_)), messages


def fit_ovr(cfg, x, y, classes, c):
    rows, bias, iterations, warning_rows = [], [], [], []
    settings = cfg.d['train']
    with stage(cfg, f'OVR C={c}', len(classes), 'classes') as bar:
        with parallel_config(backend=settings['joblib_backend'], n_jobs=settings['n_jobs'],
                             inner_max_num_threads=settings['inner_max_num_threads'], temp_folder=str(cfg.paths['joblib_temp'])):
            results = Parallel(return_as=settings['return_as'], pre_dispatch=settings['pre_dispatch'])(
                delayed(_fit_binary)(x, y, code, c, settings['svc'], cfg.d['seed']) for code in classes)
            for code, (coef, intercept, n_iter, messages) in zip(classes, results):
                rows.append(coef)
                bias.append(intercept)
                iterations.append(n_iter)
                for message in messages:
                    warning_rows.append({'class': str(code), 'message': message})
                    LOG.warning('C=%s class=%s %s', c, code, message)
                bar.update()
    return np.vstack(rows), np.asarray(bias), {'max_n_iter': max(iterations), 'warnings': warning_rows}


def score_topk(cfg, x, coef, intercept, classes, name):
    n = x.shape[0]
    k = min(cfg.d['evaluate']['top_k'], len(classes))
    top = np.empty((n, k), dtype=classes.dtype)
    step = cfg.d['evaluate']['batch_size']
    with stage(cfg, name, n) as bar:
        for start in range(0, n, step):
            end = min(start + step, n)
            scores = np.asarray(x[start:end] @ coef.T) + intercept
            # classes are sorted; stable descending scores break ties lexically.
            order = np.argsort(-scores, axis=1, kind='stable')[:, :k]
            top[start:end] = classes[order]
            bar.update(end - start)
    return top


def load_bundle(cfg):
    info = read_json(cfg.model_file('train_info'))
    assert_run(cfg, info)
    if info['splits_sha256'] != sha256(cfg, cfg.file('splits', 'splits')):
        raise ValueError('Splits changed after training; rerun training')
    with stage(cfg, 'load model', 1, 'models') as bar:
        model = joblib.load(cfg.model_file('model'))
        bar.update()
    return model, info


def main():
    args = parser(__doc__).parse_args()
    cfg = Config(args, 'train_tfidf')
    start = time.perf_counter()
    validate_inputs(cfg)
    assert_run(cfg, read_json(cfg.file('splits', 'split_info')))
    df = read_parquet(cfg, cfg.file('splits', 'splits'), 'read splits')
    validate_overlap(cfg, df)
    train_all = df.filter(pl.col('role') == cfg.d['split']['train_role'])
    with stage(cfg, 'select training label space', train_all.height) as bar:
        train, classes, class_counts = training_subset(cfg, train_all)
        bar.update(train_all.height)
    val = df.filter(pl.col('role') == cfg.d['split']['val_role'])
    if cfg.d['train']['tune_c'] and val.is_empty():
        raise ValueError('Validation set is empty; cannot tune C')
    vectorizers, x = vectorize_fit(cfg, train['text'].to_list())
    xv = transform(cfg, vectorizers, val['text'].to_list(), 'transform val') if val.height else None
    y = train['label_code'].to_numpy()
    search = cfg.d['train']['c_values'] if cfg.d['train']['tune_c'] else [cfg.d['train']['default_c']]
    best, best_score, best_c, search_rows, diagnostics = None, -np.inf, None, [], []
    for c in search:
        coef, intercept, diag = fit_ovr(cfg, x, y, classes, c)
        if val.height:
            top = score_topk(cfg, xv, coef, intercept, classes, f'validation C={c}')
            score = f1_score(val['label_code'].to_numpy(), top[:, 0], average=cfg.d['train']['validation_average'],
                             zero_division=cfg.d['train']['zero_division'])
        else:
            score = None
        search_rows.append({'C': c, 'val_macro_f1': score, 'val_records': val.height})
        diagnostics.append({'C': c, **diag})
        # First C in the configured order wins equal validation scores.
        if best is None or (score is not None and score > best_score):
            best, best_c = (coef, intercept), c
            best_score = score if score is not None else -np.inf
    majority_order = class_counts.sort(['n_texts', 'label_code'], descending=[True, False])['label_code'].to_list()
    bundle = {'vectorizers': vectorizers, 'classes': classes, 'coef': best[0], 'intercept': best[1],
              'majority_order': majority_order, 'selected_c': best_c}
    with stage(cfg, 'save fitted model', 1, 'models') as bar:
        joblib.dump(bundle, cfg.model_file('model'), compress=cfg.d['train']['joblib_compress'])
        bar.update()
    pl.DataFrame(search_rows).write_csv(cfg.file('metrics', 'c_search'))
    info = {**provenance(cfg), 'selected_c': best_c, 'train_records': train.height,
        'train_unique_texts': train['text'].n_unique(), 'train_represented_rows': int(train['n_rows'].sum()),
        'dropped_train_records': train_all.height - train.height,
        'dropped_train_codes': train_all['label_code'].n_unique() - len(classes),
        'dropped_train_rows': int(train_all['n_rows'].sum() - train['n_rows'].sum()),
        'class_count': len(classes), 'feature_count': x.shape[1], 'elapsed_seconds': time.perf_counter() - start,
        'splits_sha256': sha256(cfg, cfg.file('splits', 'splits')), 'svc_diagnostics': diagnostics,
        'sample_weight': 'uniform_one', 'majority_code': majority_order[0]}
    write_json(cfg, cfg.model_file('train_info'), info)
    LOG.info('Training complete C=%s samples=%s classes=%s features=%s seconds=%.3f',
             best_c, train.height, len(classes), x.shape[1], info['elapsed_seconds'])


if __name__ == '__main__':
    main()
