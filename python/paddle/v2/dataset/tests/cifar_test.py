import paddle.v2.dataset.cifar
import unittest


class TestCIFAR(unittest.TestCase):
    def check_reader(self, reader):
        sum = 0
        label = 0
        for l in reader():
            self.assertEqual(l[0].size, 3072)
            if l[1] > label:
                label = l[1]
            sum += 1
        return sum, label

    def test_test10(self):
        instances, max_label_value = self.check_reader(
            paddle.v2.dataset.cifar.test10())
        self.assertEqual(instances, 10000)
        self.assertEqual(max_label_value, 9)

    def test_train10(self):
        instances, max_label_value = self.check_reader(
            paddle.v2.dataset.cifar.train10())
        self.assertEqual(instances, 50000)
        self.assertEqual(max_label_value, 9)

    def test_test100(self):
        instances, max_label_value = self.check_reader(
            paddle.v2.dataset.cifar.test100())
        self.assertEqual(instances, 10000)
        self.assertEqual(max_label_value, 99)

    def test_train100(self):
        instances, max_label_value = self.check_reader(
            paddle.v2.dataset.cifar.train100())
        self.assertEqual(instances, 50000)
        self.assertEqual(max_label_value, 99)


if __name__ == '__main__':
    unittest.main()
