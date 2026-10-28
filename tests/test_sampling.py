import unittest
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from nids.sampling import StratifiedReservoir, sample_parquet


class StratifiedReservoirTests(unittest.TestCase):
    def test_caps_each_class_and_tracks_population_support(self):
        sampler = StratifiedReservoir({0: 3, 1: 2}, seed=7)
        x = np.arange(40, dtype=np.float32).reshape(20, 2)
        y = np.array([0] * 12 + [1] * 8)
        sampler.update(x[:9], y[:9])
        sampler.update(x[9:], y[9:])
        sampled_x, sampled_y, population = sampler.result()
        self.assertEqual(sampled_x.shape, (5, 2))
        self.assertEqual(np.sum(sampled_y == 0), 3)
        self.assertEqual(np.sum(sampled_y == 1), 2)
        self.assertEqual(population, {0: 12, 1: 8})

    def test_default_cap_retains_classes_not_in_initial_mapping(self):
        sampler = StratifiedReservoir({}, seed=1, default_cap=5)
        sampler.update(np.arange(20, dtype=np.float32).reshape(10, 2), ["A"] * 4 + ["B"] * 6)
        _, labels, population = sampler.result()
        self.assertEqual(set(labels), {"A", "B"})
        self.assertEqual(population, {"A": 4, "B": 6})
        self.assertEqual(np.sum(labels == "A"), 4)
        self.assertEqual(np.sum(labels == "B"), 5)

    def test_nonpositive_cap_is_rejected(self):
        with self.assertRaises(ValueError):
            StratifiedReservoir({0: 0}, seed=1)

    def test_nonfinite_training_feature_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flows.parquet"
            pq.write_table(pa.table({
                "x": [1.0, float("inf")],
                "Label": [0, 1],
                "Attack": ["Benign", "Bot"],
            }), path)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                sample_parquet(
                    str(path), feature_columns=["x"], target_column="Label",
                    class_caps={0: 1, 1: 1}, batch_size=2,
                )

    def test_missing_training_feature_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flows.parquet"
            pq.write_table(pa.table({
                "x": [1.0, None],
                "Label": [0, 1],
                "Attack": ["Benign", "Bot"],
            }), path)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                sample_parquet(
                    str(path), feature_columns=["x"], target_column="Label",
                    class_caps={0: 1, 1: 1}, batch_size=2,
                )


if __name__ == "__main__":
    unittest.main()
