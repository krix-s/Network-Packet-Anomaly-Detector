import unittest

import numpy as np
from sklearn.dummy import DummyClassifier

from nids.inference import predict_record


class InferenceValidationTests(unittest.TestCase):
    def setUp(self):
        model = DummyClassifier(strategy="most_frequent").fit([[0], [1]], [0, 1])
        self.artifact = {
            "task": "binary",
            "target": "Label",
            "features": ["x"],
            "classes": [0, 1],
            "model": model,
        }

    def test_missing_feature_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing required"):
            predict_record(self.artifact, {})

    def test_non_numeric_and_non_finite_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be numeric"):
            predict_record(self.artifact, {"x": "bad"})
        with self.assertRaisesRegex(ValueError, "must be finite"):
            predict_record(self.artifact, {"x": np.inf})

    def test_boolean_is_not_accepted_as_a_numeric_feature(self):
        with self.assertRaisesRegex(ValueError, "not boolean"):
            predict_record(self.artifact, {"x": True})

    def test_prediction_returns_task_and_probabilities(self):
        result = predict_record(self.artifact, {"x": 1.0})
        self.assertEqual(result["task"], "binary")
        self.assertEqual(result["target"], "Label")
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0)

    def test_values_follow_artifact_feature_order(self):
        class CapturingModel:
            classes_ = np.asarray([0, 1])

            def predict(self, x):
                self.seen = x.copy()
                return np.asarray([1])

            def predict_proba(self, x):
                return np.asarray([[0.2, 0.8]])

        model = CapturingModel()
        artifact = {**self.artifact, "features": ["second", "first"], "model": model}
        predict_record(artifact, {"first": 1.0, "second": 2.0})
        np.testing.assert_array_equal(model.seen, [[2.0, 1.0]])


if __name__ == "__main__":
    unittest.main()
