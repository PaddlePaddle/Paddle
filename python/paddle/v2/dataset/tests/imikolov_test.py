import paddle.v2.dataset.imikolov
import unittest


class TestMikolov(unittest.TestCase):
    def check_reader(self, reader, n):
        for l in reader():
            self.assertEqual(len(l), n)

    def test_train(self):
        n = 5
        self.check_reader(paddle.v2.dataset.imikolov.train(n), n)

    def test_test(self):
        n = 5
        self.check_reader(paddle.v2.dataset.imikolov.test(n), n)


if __name__ == '__main__':
    unittest.main()
