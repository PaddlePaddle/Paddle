import paddle.v2.dataset.imikolov
import unittest

WORD_DICT = paddle.v2.dataset.imikolov.build_dict()


class TestMikolov(unittest.TestCase):
    def check_reader(self, reader, n):
        for l in reader():
            self.assertEqual(len(l), n)

    def test_train(self):
        n = 5
        self.check_reader(paddle.v2.dataset.imikolov.train(WORD_DICT, n), n)

    def test_test(self):
        n = 5
        self.check_reader(paddle.v2.dataset.imikolov.test(WORD_DICT, n), n)

    def test_total(self):
        _, idx = zip(*WORD_DICT.items())
        self.assertEqual(sorted(idx)[-1], len(WORD_DICT) - 1)


if __name__ == '__main__':
    unittest.main()
