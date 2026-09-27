import unittest

from train import load_dataset


class DatasetTests(unittest.TestCase):
    def test_dataset_shape_and_positive_class(self) -> None:
        features, target, feature_names = load_dataset()

        self.assertEqual(features.shape, (569, 30))
        self.assertEqual(len(feature_names), 30)
        self.assertEqual(int(target.sum()), 212)
        self.assertEqual(int((target == 0).sum()), 357)


if __name__ == "__main__":
    unittest.main()