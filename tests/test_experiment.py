import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from nids.experiment import _fit_sample_weights, _splits, run_experiment
from nids.inference import load_artifact, predict_record


class EndToEndExperimentTests(unittest.TestCase):
    def test_source_prior_weights_restore_population_proportions(self):
        y = np.array(["common", "common", "rare", "rare"])
        # The capped sample is balanced, while the source population is 9:1.
        population_weights = np.array([9.0, 9.0, 1.0, 1.0])
        weights = _fit_sample_weights("source-prior", y, population_weights)
        self.assertAlmostEqual(weights[:2].sum() / weights.sum(), 0.9)
        self.assertAlmostEqual(weights.mean(), 1.0)

    def test_source_prior_weights_validate_shape_and_strategy(self):
        y = np.array([0, 1])
        with self.assertRaisesRegex(ValueError, "match labels"):
            _fit_sample_weights("source-prior", y, np.array([1.0]))
        with self.assertRaisesRegex(ValueError, "training_weighting"):
            _fit_sample_weights("unknown", y, np.ones(2))

    def test_stratified_split_is_reproducible_disjoint_and_70_15_15(self):
        y = np.repeat([0, 1], 100)
        first = _splits(y)
        second = _splits(y)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
        train, validation, test = first
        self.assertEqual((len(train), len(validation), len(test)), (140, 30, 30))
        self.assertFalse(set(train) & set(validation))
        self.assertFalse(set(train) & set(test))
        self.assertFalse(set(validation) & set(test))

    def test_binary_and_multiclass_save_reload_and_predict(self):
        rng = np.random.default_rng(12)
        attacks = np.repeat(["Benign", "Bot", "SQL"], 60)
        labels = (attacks != "Benign").astype(np.int8)
        signal = labels.astype(np.float32) + rng.normal(0, 0.2, len(labels))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "toy.parquet"
            models = root / "models"
            reports = root / "reports"
            pq.write_table(pa.table({
                "signal": signal,
                "aux": rng.normal(size=len(labels)).astype(np.float32),
                "Label": labels,
                "Attack": attacks,
            }), dataset)

            for task in ("binary", "multiclass"):
                summary = run_experiment(
                    task,
                    data_path=dataset,
                    output_dir=models,
                    report_dir=reports,
                    multiclass_cap=50,
                    batch_size=30,
                )
                self.assertTrue((reports / f"{task}_metrics.json").is_file())
                artifact = load_artifact(models / f"{task}_model.joblib")
                record = {name: float(signal[0] if name == "signal" else 0.0) for name in artifact["features"]}
                prediction = predict_record(artifact, record)
                self.assertEqual(prediction["task"], task)
                self.assertIn("confidence", prediction)
                self.assertTrue(summary["test_metrics"])


if __name__ == "__main__":
    unittest.main()
