import paddle.v2.dataset.common
import unittest
import tempfile


class TestCommon(unittest.TestCase):
    def test_md5file(self):
        _, temp_path = tempfile.mkstemp()
        with open(temp_path, 'w') as f:
            f.write("Hello\n")
        self.assertEqual('09f7e02f1290be211da707a266f153b3',
                         paddle.v2.dataset.common.md5file(temp_path))

    def test_download(self):
        yi_avatar = 'https://avatars0.githubusercontent.com/u/1548775?v=3&s=460'
        self.assertEqual(
            paddle.v2.dataset.common.DATA_HOME + '/test/1548775?v=3&s=460',
            paddle.v2.dataset.common.download(
                yi_avatar, 'test', 'f75287202d6622414c706c36c16f8e0d'))


if __name__ == '__main__':
    unittest.main()
