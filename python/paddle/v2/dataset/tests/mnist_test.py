import paddle.v2.dataset.mnist
import unittest


class TestMNIST(unittest.TestCase):
    def check_reader(self, reader):
        sum = 0
        label = 0
        for l in reader():
            self.assertEqual(l[0].size, 784)
            if l[1] > label:
                label = l[1]
            sum += 1
        return sum, label

    def test_train(self):
        instances, max_label_value = self.check_reader(
            paddle.v2.dataset.mnist.train())
        self.assertEqual(instances, 60000)
        self.assertEqual(max_label_value, 9)

    def test_test(self):
        instances, max_label_value = self.check_reader(
            paddle.v2.dataset.mnist.test())
        self.assertEqual(instances, 10000)
        self.assertEqual(max_label_value, 9)


if __name__ == '__main__':
    unittest.main()
