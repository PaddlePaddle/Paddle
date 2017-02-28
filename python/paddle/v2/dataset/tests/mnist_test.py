import paddle.v2.dataset.mnist
import unittest

class TestMNIST(unittest.TestCase):
    def check_reader(self, reader):
        sum = 0
        for l in reader:
            self.assertEqual(l[0].size, 784)
            self.assertEqual(l[1].size, 1)
            self.assertLess(l[1], 10)
            self.assertGreaterEqual(l[1], 0)
            sum += 1
        return sum

    def test_train(self):
        self.assertEqual(
            self.check_reader(paddle.v2.dataset.mnist.train()),
            60000)

    def test_test(self):
        self.assertEqual(
            self.check_reader(paddle.v2.dataset.mnist.test()),
            10000)


if __name__ == '__main__':
    unittest.main()
