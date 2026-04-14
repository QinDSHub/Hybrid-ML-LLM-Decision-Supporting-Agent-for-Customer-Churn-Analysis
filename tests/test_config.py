from pathlib import Path
import unittest

from churn_retrieval.config import load_config


class ConfigTests(unittest.TestCase):
    def test_load_config_resolves_project_paths(self):
        project_root = Path(__file__).resolve().parents[1]
        config = load_config(project_root / "configs" / "default.toml")
        self.assertEqual(config.paths.project_root, project_root)
        self.assertEqual(config.paths.raw_data_dir, project_root / "data" / "raw")
        self.assertEqual(config.model.top_k, 10)
        self.assertEqual(config.evaluation.prediction_path, project_root / "outputs" / "predictions" / "prediction.csv")


if __name__ == "__main__":
    unittest.main()
