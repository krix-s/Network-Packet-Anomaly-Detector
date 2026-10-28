import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from nids.experiment import run_experiment
from nids.inference import load_artifact, predict_record


class EndToEndExperimentTests(unittest.TestCase):
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
