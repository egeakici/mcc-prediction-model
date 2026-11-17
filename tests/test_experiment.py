"""Independent checks for units, cluster weighting and leakage boundaries."""
import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import polars as pl
import yaml
from sklearn.metrics import f1_score

from src.config import Config
from src.evaluate import bootstrap, f1_from_counts, merchant_arrays, merchant_table, metrics
from src.make_splits import fold_for_key
from src.text_norm import normalize_text, text_expr
from src.train_tfidf import training_subset


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = yaml.safe_load(Path('configs/config.yaml').read_text(encoding='utf-8'))
        cls.cfg = SimpleNamespace(d=cls.d)

    def test_fold_matches_exact_specification(self):
        for key in ['WALMART', 'GRAINGER', 'İSTANBUL', '0763']:
            expected = int(hashlib.sha1(key.encode('utf-8')).hexdigest(), 16) % self.d['split']['n_folds']
            self.assertEqual(fold_for_key(key, self.cfg), expected)

    def test_text_normalization_has_no_intermediary(self):
        df = pl.DataFrame({'key_strict': ['  ACME  STORE\tLTD  '], 'intermediary': ['SECRET']})
        expected = 'acme store ltd'
        self.assertEqual(normalize_text(df['key_strict'][0], self.cfg, 'SECRET'), expected)
        self.assertEqual(df.select(text_expr(self.cfg))['text'][0], expected)

    def test_merchant_aggregation_sums_fragmented_labels_and_texts(self):
        # a's text x has 6+6=12 rows versus y's 10; individual records give the wrong winner.
        df = pl.DataFrame({'key_loose': ['a', 'a', 'a', 'b'], 'text': ['x', 'x', 'y', 'z'],
            'label_code': ['1000', '2000', '2000', '3000'], 'pred_code': ['1000', '1000', '2000', '1000'],
            'n_rows': [6, 6, 10, 2]})
        merchant = merchant_table(df)
        a = merchant.filter(pl.col('key_loose') == 'a').row(0, named=True)
        self.assertEqual(a['label_code'], '2000')
        self.assertEqual(a['label_rows'], 16)
        self.assertEqual(a['text'], 'x')
        self.assertEqual(a['pred_code'], '1000')
        self.assertEqual(a['correct_rows'], 16)

    def test_unique_text_threshold_does_not_count_origins_or_rows(self):
        n = self.d['train']['min_texts_per_class']
        rows = [{'text': f'{code}_{i}', 'label_code': code, 'n_rows': 100}
                for code in ['1000', '2000'] for i in range(n)]
        rows += [{'text': 'same', 'label_code': '3000', 'n_rows': 99999} for _ in range(n)]
        selected, classes, _ = training_subset(self.cfg, pl.DataFrame(rows))
        self.assertEqual(classes.tolist(), ['1000', '2000'])
        self.assertEqual(selected.height, 2 * n)

    def test_f1_matches_sklearn_with_cluster_multiplicities_and_oov(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        p = np.array([0, 1, 1, 0, 0, 1])
        mult = np.array([3, 0, 2, 1, 1, 2])
        eligible = np.array([0, 1])
        expected = f1_score(np.repeat(y, mult), np.repeat(p, mult), labels=eligible, average='macro', zero_division=0)
        self.assertAlmostEqual(f1_from_counts(y, p, eligible, mult, 3), expected)

    def test_metric_units_and_cluster_bootstrap(self):
        # Five merchants per class; each merchant has one correctly/incorrectly predicted text.
        n = self.d['evaluate']['min_merchants_per_class']
        rows = []
        for code, pred, weight in [('1000', '1000', 9), ('2000', '1000', 1)]:
            for i in range(n):
                rows.append({'key_loose': f'{code}_{i}', 'text': f'{code}_{i}', 'label_code': code,
                    'pred_code': pred, 'label_group': 'g', 'pred_group': 'g', 'label_origin': 'original',
                    'n_rows': weight, 'top3_codes': ['1000', '2000'], 'seen_in_ok': False})
        df = pl.DataFrame(rows)
        result, report = metrics(self.cfg, df, ['1000', '2000'], {}, cross_source=True)
        self.assertAlmostEqual(result['acc_rows'], 0.9)
        self.assertAlmostEqual(result['acc_texts'], 0.5)
        self.assertAlmostEqual(result['acc_merchant'], 0.5)
        self.assertAlmostEqual(result['macro_f1'], 1 / 3)
        self.assertEqual(result['macro_f1_n_classes'], 2)
        self.assertEqual(result['ceiling_rows'], 1.0)
        self.assertEqual(result['top3_rows'], 1.0)
        self.assertEqual(result['same_group_error_share'], 1.0)
        self.assertIsNone(result['acc_rows_origin_linked'])
        self.assertIsNone(result['acc_rows_seen_in_ok_true'])
        self.assertEqual(report['support'].sum(), 2 * n)
        merchants = merchant_table(df)
        repeat = bootstrap(self.cfg, merchants, merchant_arrays(self.cfg, merchants))
        for key, value in repeat.items():
            self.assertEqual(result[key], value)

    def test_output_path_guard(self):
        cfg = Config(argparse.Namespace(config=Path('configs/config.yaml'), sample=20000), 'tests')
        for path in [cfg.paths['raw'] / 'x.parquet', cfg.paths['reference'] / 'x.csv', cfg.root.parent / 'x']:
            with self.assertRaises(ValueError):
                cfg.guard(path)


if __name__ == '__main__':
    unittest.main()
