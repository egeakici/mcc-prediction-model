"""Cross-check actual smoke artifacts without rerunning any training."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import unittest
import joblib
import numpy as np
import polars as pl
from sklearn.metrics import f1_score
import yaml


class SmokeArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parent.parent
        cls.config = yaml.safe_load((cls.root / 'configs/config.yaml').read_text(encoding='utf-8'))
        cls.paths = {k: cls.root / v for k, v in cls.config['sample_paths'].items()}
        split_path = cls.paths['splits'] / cls.config['files']['splits']
        if not split_path.exists():
            raise unittest.SkipTest('Smoke artifacts not built yet')
        cls.splits = pl.read_parquet(split_path)

    def test_final_config_matches_artifacts(self):
        expected = hashlib.sha256(yaml.safe_dump(self.config, sort_keys=True).encode('utf-8')).hexdigest()
        for path in [self.paths['interim'] / 'build_info.json', self.paths['splits'] / 'split_info.json',
                     self.paths['models'] / 'tfidf_linearsvc/train_info.json']:
            info = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(info['config_sha256'], expected)

    def test_all_folds_roles_and_anchor_flags(self):
        train = set(self.splits.filter(pl.col('role') == 'train')['key_loose'])
        anchor = set(self.splits.filter(pl.col('source_id') == 'oklahoma')['key_loose'])
        for row in self.splits.iter_rows(named=True):
            fold = int(hashlib.sha1(row['key_loose'].encode('utf-8')).hexdigest(), 16) % 10
            self.assertEqual(row['fold'], fold)
            self.assertEqual(row['seen_in_ok'], row['key_loose'] in anchor)
            expected_role = 'unused'
            for name, spec in self.config['split']['roles'].items():
                if spec['source'] == row['source_id'] and fold in spec['folds']:
                    expected_role = name
            self.assertEqual(row['role'], expected_role)
            if row['role'] in self.config['evaluate']['test_sets']:
                self.assertNotIn(row['key_loose'], train)

    def test_raw_reference_hashes_unchanged(self):
        info = json.loads((self.paths['interim'] / 'build_info.json').read_text(encoding='utf-8'))
        for name, expected in info['input_sha256'].items():
            with (self.root / name).open('rb') as handle:
                self.assertEqual(hashlib.file_digest(handle, 'sha256').hexdigest(), expected)

    def test_exclusion_accounting_and_no_code_rows(self):
        summary = pl.read_csv(self.paths['interim'] / 'table_summary.csv')
        exclusions = pl.read_csv(self.paths['interim'] / 'exclusions.csv')
        for row in summary.iter_rows(named=True):
            self.assertEqual(row['raw_rows'], 20000)
            self.assertEqual(row['remaining_rows'] + row['excluded_rows'], row['raw_rows'])
            self.assertEqual(exclusions.filter(pl.col('source_id') == row['source_id'])['n_rows'].sum(), row['excluded_rows'])
            no_code = pl.read_parquet(self.paths['interim'] / f"no_code_{row['source_id']}.parquet")
            self.assertEqual(no_code.height, row['no_code_rows'])
            self.assertTrue(all(not x for x in no_code['mcc_analysis'].fill_null('')))

    def test_model_uses_eligible_training_classes_and_only_train_vocabulary(self):
        bundle = joblib.load(self.paths['models'] / 'tfidf_linearsvc/model.joblib')
        train = self.splits.filter(pl.col('role') == 'train')
        texts = defaultdict(set)
        for text, code in train.select('text', 'label_code').iter_rows():
            texts[code].add(text)
        expected = sorted(c for c in texts if len(texts[c]) >= self.config['train']['min_texts_per_class'])
        self.assertEqual(bundle['classes'].tolist(), expected)
        kept_texts = train.filter(pl.col('label_code').is_in(expected))['text'].to_list()
        for _, vectorizer in bundle['vectorizers']:
            analyzer = vectorizer.build_analyzer()
            df_counts = Counter(term for text in kept_texts for term in set(analyzer(text)))
            expected_vocabulary = {term for term, count in df_counts.items() if count >= vectorizer.min_df}
            self.assertEqual(set(vectorizer.vocabulary_), expected_vocabulary)
        self.assertEqual(bundle['coef'].shape[0], len(expected))
        self.assertEqual(bundle['coef'].shape[1], sum(len(v.vocabulary_) for _, v in bundle['vectorizers']))

    def test_metrics_against_independent_python_calculation(self):
        results = pl.read_csv(self.paths['metrics'] / 'results.csv')
        for test in self.config['evaluate']['test_sets']:
            for model in self.config['train']['model_names']:
                df = pl.read_parquet(self.paths['predictions'] / f'{model}_{test}.parquet')
                metrics = dict(results.filter((pl.col('model') == model) & (pl.col('test_set') == test)).select('metric', 'value').iter_rows())
                total = sum(df['n_rows'])
                correct = sum(r['n_rows'] for r in df.iter_rows(named=True) if r['pred_code'] == r['label_code'])
                self.assertAlmostEqual(metrics['acc_rows'], correct / total)
                pairs = {(r['text'], r['label_code'], r['pred_code']) for r in df.iter_rows(named=True)}
                self.assertAlmostEqual(metrics['acc_texts'], sum(t == p for _, t, p in pairs) / len(pairs))
                groups = defaultdict(list)
                for row in df.iter_rows(named=True):
                    groups[row['key_loose']].append(row)
                ys, ps, ceiling = [], [], 0
                for key in sorted(groups):
                    labels, texts, predictions = Counter(), Counter(), {}
                    for row in groups[key]:
                        labels[row['label_code']] += row['n_rows']
                        texts[row['text']] += row['n_rows']
                        predictions[row['text']] = row['pred_code']
                    true = sorted(labels, key=lambda x: (-labels[x], x))[0]
                    chosen = sorted(texts, key=lambda x: (-texts[x], x))[0]
                    ys.append(true)
                    ps.append(predictions[chosen])
                    ceiling += labels[true]
                self.assertAlmostEqual(metrics['acc_merchant'], np.mean(np.array(ys) == np.array(ps)))
                eligible = sorted(c for c, count in Counter(ys).items() if count >= self.config['evaluate']['min_merchants_per_class'])
                expected = f1_score(ys, ps, labels=eligible, average='macro', zero_division=0)
                self.assertAlmostEqual(metrics['macro_f1'], expected)
                self.assertEqual(metrics['macro_f1_n_classes'], len(eligible))
                self.assertAlmostEqual(metrics['ceiling_rows'], ceiling / total)
                for name in ['acc_rows', 'macro_f1']:
                    lo, hi = metrics[name + '_ci_low'], metrics[name + '_ci_high']
                    self.assertLessEqual(0, lo)
                    self.assertLessEqual(lo, hi)
                    self.assertLessEqual(hi, 1)

    def test_leakage_sample_and_report(self):
        leakage = pl.read_csv(self.paths['metrics'] / 'leakage.csv')
        row = leakage.filter(pl.col('protocol') == 'random_raw_row_split').row(0, named=True)
        self.assertEqual(row['sample_rows'], self.config['leakage']['smoke_sample_rows'])
        self.assertEqual(row['test_rows'], self.config['leakage']['smoke_sample_rows'] * self.config['leakage']['test_size'])
        self.assertGreater(row['test_rows_seen_in_train_share'], 0)
        report = (self.paths['reports'] / self.config['files']['report']).read_text(encoding='utf-8')
        self.assertEqual(report.count('## '), 16)
        for test in self.config['evaluate']['test_sets']:
            self.assertIn(test, report)
        self.assertIn('Sızıntı kontrolü', report)


if __name__ == '__main__':
    unittest.main()
