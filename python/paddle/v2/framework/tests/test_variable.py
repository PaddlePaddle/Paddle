import unittest
from paddle.v2.framework.graph import Variable
import paddle.v2.framework.core as core
import numpy as np


class TestVariable(unittest.TestCase):
    def test_np_dtype_convert(self):
        DT = core.DataType
        convert = Variable._convert_np_dtype_to_dtype_
        self.assertEqual(DT.FP32, convert(np.float32))
        self.assertEqual(DT.FP16, convert("float16"))
        self.assertEqual(DT.FP64, convert("float64"))
        self.assertEqual(DT.INT32, convert("int32"))
        self.assertEqual(DT.INT16, convert("int16"))
        self.assertEqual(DT.INT64, convert("int64"))
        self.assertEqual(DT.BOOL, convert("bool"))
        self.assertRaises(ValueError, lambda: convert("int8"))


if __name__ == '__main__':
    unittest.main()
