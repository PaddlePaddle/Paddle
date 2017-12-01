import unittest
from paddle.v2.fluid.framework import default_main_program, Program
import paddle.v2.fluid.dtypes as dtypes
import paddle.v2.fluid.core as core
import numpy as np


class TestVariable(unittest.TestCase):
    def test_np_dtype_convert(self):
        self.assertEqual(dtypes.float32, as_dtype(np.float32))
        self.assertEqual(dtypes.bool, as_dtype("bool"))
        self.assertEqual(dtypes.int16, as_dtype("int16"))
        self.assertEqual(dtypes.int32, as_dtype("int32"))
        self.assertEqual(dtypes.int64, as_dtype("int64"))
        self.assertEqual(dtypes.float16, as_dtype("float16"))
        self.assertEqual(dtypes.float32, as_dtype("float32"))
        self.assertEqual(dtypes.float64, as_dtype("float64"))
        self.assertRaises(ValueError, as_dtype("int8"))
        self.assertRaises(ValueError, as_dtype(np.int8))

    def test_var(self):
        b = default_main_program().current_block()
        w = b.create_var(
            dtype="float64", shape=[784, 100], lod_level=0, name="fc.w")
        self.assertNotEqual(str(w), "")
        self.assertEqual(core.DataType.FP64, w.dtype)
        self.assertEqual((784, 100), w.shape)
        self.assertEqual("fc.w", w.name)
        self.assertEqual(0, w.lod_level)

        w = b.create_var(name='fc.w')
        self.assertEqual(core.DataType.FP64, w.dtype)
        self.assertEqual((784, 100), w.shape)
        self.assertEqual("fc.w", w.name)
        self.assertEqual(0, w.lod_level)

        self.assertRaises(ValueError,
                          lambda: b.create_var(name="fc.w", shape=(24, 100)))

    def test_step_scopes(self):
        prog = Program()
        b = prog.current_block()
        var = b.create_var(
            name='step_scopes', type=core.VarDesc.VarType.STEP_SCOPES)
        self.assertEqual(core.VarDesc.VarType.STEP_SCOPES, var.type)


if __name__ == '__main__':
    unittest.main()
