import unittest

from churn_retrieval.modeling.feature_builder import car_level_to_text
from churn_retrieval.preprocessing.cleaning import normalize_repair_type


class HelperTests(unittest.TestCase):
    def test_normalize_repair_type(self):
        value = normalize_repair_type("首保;普修A;普修A;普修B")
        self.assertEqual(value, "普通维修;首次保养")

    def test_car_level_to_text(self):
        self.assertEqual(car_level_to_text("family_1"), "高档车")
        self.assertEqual(car_level_to_text("family_2"), "中档车")
        self.assertEqual(car_level_to_text("other"), "低档车")


if __name__ == "__main__":
    unittest.main()
