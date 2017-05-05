import paddle.v2.dataset.ptb
import unittest

WORD_DICT = paddle.v2.dataset.ptb.build_dict()


class TestMikolov(unittest.TestCase):
    def check_reader(self, reader, n):
        for l in reader():
            self.assertEqual(len(l), n)

    def test_ngram_train(self):
        n = 5
        self.check_reader(paddle.v2.dataset.ptb.ngram_train(WORD_DICT, n), n)

    def test_ngram_test(self):
        n = 5
        self.check_reader(paddle.v2.dataset.ptb.ngram_test(WORD_DICT, n), n)

    def test_seq_train(self):
        first_line = 'aer banknote berlitz calloway centrust cluett fromstein '\
                'gitano guterman hydro-quebec ipo kia memotec mlx nahb punts '\
                'rake regatta rubens sim snack-food ssangyong swapo wachter'
        first_line = [
            WORD_DICT.get(ch, WORD_DICT['<unk>'])
            for ch in first_line.split(' ')
        ]
        for l in paddle.v2.dataset.ptb.seq_train(WORD_DICT)():
            read_line = l[0][1:]
            break

        self.assertEqual(first_line, read_line)

    def test_seq_test(self):
        first_line = 'consumers may want to move their telephones a little '\
                'closer to the tv set'
        first_line = [
            WORD_DICT.get(ch, WORD_DICT['<unk>'])
            for ch in first_line.split(' ')
        ]
        for l in paddle.v2.dataset.ptb.seq_test(WORD_DICT)():
            read_line = l[0][1:]
            break

        self.assertEqual(first_line, read_line)

    def test_total(self):
        _, idx = zip(*WORD_DICT.items())
        self.assertEqual(sorted(idx)[-1], len(WORD_DICT) - 1)


if __name__ == '__main__':
    unittest.main()
