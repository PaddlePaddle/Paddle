import paddle.v2.dataset.common
import unittest
import tempfile

class TestCommon(unittest.TestCase):
    def test_md5file(self):
        _, temp_path =tempfile.mkstemp()
        f = open(temp_path, 'w')
        f.write("Hello\n")
        f.close()
        self.assertEqual(
            '09f7e02f1290be211da707a266f153b3',
            paddle.v2.dataset.common.md5file(temp_path))

if __name__ == '__main__':
    unittest.main()
