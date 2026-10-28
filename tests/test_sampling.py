import unittest

import numpy as np

from nids.sampling import StratifiedReservoir


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


if __name__ == "__main__":
    unittest.main()
