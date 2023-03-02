import unittest

import numpy as np

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import Program, program_guard


class TestGreaterEqualOpFp16(unittest.TestCase):
    def test_api_fp16(self):
        paddle.enable_static()
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            label = paddle.to_tensor([3, 3], dtype="float16")
            limit = paddle.to_tensor([3, 2], dtype="float16")
            out = paddle.greater_equal(x=label, y=limit)
            if core.is_compiled_with_cuda():
                place = paddle.CUDAPlace(0)
                exe = fluid.Executor(place)
                (res,) = exe.run(fetch_list=[out])
                self.assertEqual((res == np.array([True, True])).all(), True)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
