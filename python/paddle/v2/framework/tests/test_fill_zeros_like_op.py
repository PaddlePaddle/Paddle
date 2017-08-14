import unittest
from op_test_util import OpTestMeta
import numpy


class TestFillZerosLikeOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "fill_zeros_like"
        self.inputs = {'Src': numpy.random.random((29, 22)).astype("float32")}
        self.outputs = {'Dst': numpy.zeros_like(self.inputs['Src'])}


if __name__ == '__main__':
    unittest.main()
