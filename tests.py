import unittest
from pyspark import SparkContext, SparkConf
from look_alike import calculate_ratings


class LookAlikeTest(unittest.TestCase):
    def setUp(self):
        conf = SparkConf().setAppName("Tests").setMaster("local")
        self.sc = SparkContext(conf=conf)

    def tearDown(self):
        self.sc.stop()

    def test_ratings_calculation(self):
        data = [("u1", 123), ("u1", 123), ("u1", 132),
                ("u2", 123), ("u2", 111), ("u2", 111), ("u2", 111), ("u2", 111),
                ("u3", 123), ("u3", 123), ("u3", 125), ("u3", 125), ("u3", 111)]
        input_data = self.sc.parallelize(data)
        ratings = calculate_ratings(input_data).collectAsMap()
        self.assertEqual(ratings["u1"][123], 1.0)
        self.assertEqual(ratings["u1"][132], 0.5)
        self.assertEqual(ratings["u2"][111], 1.0)
        self.assertEqual(ratings["u2"][123], 0.25)
        self.assertEqual(ratings["u3"][123], 1.0)
        self.assertEqual(ratings["u3"][125], 1.0)
        self.assertEqual(ratings["u3"][111], 0.5)

if __name__ == '__main__':
    unittest.main()