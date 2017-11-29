import unittest
from paddle.v2.fluid.framework import default_main_program
import paddle.v2.fluid.core as core
from paddle.v2.fluid.executor import Executor
import paddle.v2.fluid.io as io
from paddle.v2.fluid.initializer import ConstantInitializer
import numpy as np

main_program = default_main_program()


class TestParameter(unittest.TestCase):
    def test_param(self):
        shape = [784, 100]
        val = 1.0625
        b = main_program.global_block()
        param = b.create_parameter(
            name='fc.w',
            shape=shape,
            dtype='float32',
            initializer=ConstantInitializer(val))
        self.assertIsNotNone(param)
        self.assertEqual('fc.w', param.name)
        self.assertEqual((784, 100), param.shape)
        self.assertEqual(core.DataType.FP32, param.dtype)
        self.assertEqual(0, param.block.idx)
        exe = Executor(core.CPUPlace())
        p = exe.run(main_program, fetch_list=[param])[0]
        self.assertTrue(np.allclose(p, np.ones(shape) * val))
        p = io.get_parameter_value_by_name('fc.w', exe, main_program)
        self.assertTrue(np.allclose(np.array(p), np.ones(shape) * val))


if __name__ == '__main__':
    unittest.main()
