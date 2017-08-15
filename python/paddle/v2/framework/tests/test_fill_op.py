import unittest
from op_test_util import OpTestMeta
import numpy


class TestFillOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "fill"
        data = [0.1, 0.2, 0.3, 0.4]

        self.attrs = {'data': data, 'shape': [2, 2], 'run_once': True}
        self.outputs = {
            'Out': numpy.array(
                [[0.1, 0.2], [0.3, 0.4]], dtype=numpy.float32)
        }


if __name__ == '__main__':
    unittest.main()
