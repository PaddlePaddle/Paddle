#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import unittest
import numpy as np
import paddle.fluid.core as core
from op_test import OpTest
import paddle
import paddle.fluid as fluid
import paddle.nn.functional as F


def ref_celu(x, alpha=1.5):
    data = alpha * (np.exp(x / alpha) - 1)
    return np.where(x <= 0, 0, x) + np.where(data >= 0, 0, data)


class TestCeluAPI(unittest.TestCase):
    # test paddle.nn.CELU, paddle.nn.functional.celu
    def setUp(self):
        self.alpha = 2.0
        self.x_np = np.random.normal(size=[3, 5, 5, 10]).astype(np.float64)

        self.place=paddle.CUDAPlace(0) if core.is_compiled_with_cuda() \
            else paddle.CPUPlace()

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', self.x_np.shape, self.x_np.dtype)
            out1 = F.celu(x, self.alpha)
            celu = paddle.nn.CELU(self.alpha)
            out2 = celu(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
        out_ref = ref_celu(self.x_np, self.alpha)
        for r in res:
            self.assertEqual(np.allclose(out_ref, r), True)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out1 = F.celu(x, self.alpha)
        celu = paddle.nn.CELU(self.alpha)
        out2 = celu(x)
        out_ref = ref_celu(self.x_np, self.alpha)
        for r in [out1, out2]:
            self.assertEqual(np.allclose(out_ref, r.numpy()), True)
        paddle.enable_static()

    def test_errors(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, F.celu, 1)
            # The input dtype must be float16, float32, float64.
            x_int32 = paddle.fluid.data(
                name='x_int32', shape=[12, 10], dtype='int32')
            self.assertRaises(TypeError, F.celu, x_int32)
            # The alpha must be not equal 0
            x_fp32 = paddle.fluid.data(
                name='x_fp32', shape=[12, 10], dtype='float32')
            self.assertRaises(ZeroDivisionError, F.celu, x_fp32, 0)
            # support the input dtype is float16
            x_fp16 = paddle.fluid.data(
                name='x_fp16', shape=[12, 10], dtype='float16')
            F.celu(x_fp16)


if __name__ == "__main__":
    unittest.main()
