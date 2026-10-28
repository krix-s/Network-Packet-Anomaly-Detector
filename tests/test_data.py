import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from nids.data import iter_parquet_batches, parquet_metadata
from nids.features import feature_columns


class ParquetDataTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tempdir.name) / "flows.parquet"
        pq.write_table(
            pa.table({"Label": [0, 1], "Attack": ["Benign", "Bot"], "feature": [1.0, 2.0]}),
            self.path,
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def test_metadata_does_not_require_dataframe_loading(self):
        result = parquet_metadata(self.path)
        self.assertEqual(result["rows"], 2)
        self.assertIn("Attack", result["columns"])

    def test_batches_allow_column_projection(self):
        batches = list(iter_parquet_batches(self.path, columns=["Label", "Attack"], batch_size=1))
        self.assertEqual([batch.num_rows for batch in batches], [1, 1])
        self.assertEqual(batches[0].schema.names, ["Label", "Attack"])

    def test_invalid_batch_size_is_rejected(self):
        with self.assertRaises(ValueError):
            list(iter_parquet_batches(self.path, batch_size=0))

    def test_missing_target_column_is_rejected(self):
        bad_path = Path(self.tempdir.name) / "bad.parquet"
        pq.write_table(pa.table({"Label": [0]}), bad_path)
        with self.assertRaisesRegex(ValueError, "target columns"):
            list(iter_parquet_batches(bad_path))

    def test_feature_selection_never_returns_either_target(self):
        columns = ["feature", "Label", "Attack", "FTP_COMMAND_RET_CODE"]
        self.assertEqual(feature_columns(columns, target="Label"), ["feature"])
        self.assertEqual(feature_columns(columns, target="Attack"), ["feature"])

    def test_feature_selection_rejects_unknown_target(self):
        with self.assertRaisesRegex(ValueError, "Unsupported target"):
            feature_columns(["feature", "Label"], target="feature")


if __name__ == "__main__":
    unittest.main()
