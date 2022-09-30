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

import unittest
import numpy as np
import paddle.fluid.core as core
from op_test import OpTest
import paddle
import paddle.fluid as fluid
import paddle.incubate as incubate

paddle.enable_static()


def _get_softmax_upper(x, fp16=True):
    x_lower = np.tril(x)
    masked_x = np.where(x_lower == 0, -10000.0, x_lower).astype("float32")
    max_value = np.max(masked_x, axis=-1, keepdims=True)
    before_exp = masked_x - max_value
    exp = np.exp(before_exp)
    exp_sum = np.sum(exp, axis=-1, keepdims=True)
    rst = exp / exp_sum
    if fp16:
        rst = rst.astype("float16")
    return rst


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestSoftmaxMaskFuseOp(OpTest):

    def setUp(self):
        self.op_type = "fused_softmax_mask_upper_triangle"
        x = np.random.random((1, 4, 32, 32)).astype("float16")
        self.inputs = {'X': x}
        rst = _get_softmax_upper(x)
        self.outputs = {'Out': rst}

    def test_check_output(self):
        self.check_output_with_place(core.CUDAPlace(0))

    def test_check_grad(self):
        self.check_grad_with_place(core.CUDAPlace(0), ["X"], "Out")


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestSoftmaxMaskFuseOp1(OpTest):

    def setUp(self):
        self.op_type = "fused_softmax_mask_upper_triangle"
        x = np.random.random((1, 4, 32, 32))
        self.inputs = {'X': x}
        rst = _get_softmax_upper(x)
        self.outputs = {'Out': rst}

    def test_check_output(self):
        try:
            self.check_output_with_place(core.CPUPlace())
        except NotImplementedError:
            pass

    def test_check_grad(self):
        try:
            self.check_grad_with_place(core.CPUPlace(), ["X"], "Out")
        except NotImplementedError:
            pass


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestDropoutBiasFuseOp2(unittest.TestCase):
    # test the python side API for softmax_mask_fuse op
    def setUp(self):
        np.random.seed(123)
        self.dtypes = ['float16', 'float32']

    def test_static(self):
        for dtype in self.dtypes:
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                input_x = fluid.data(name="x",
                                     shape=[1, 4, 32, 32],
                                     dtype=dtype)
                rst = incubate.softmax_mask_fuse_upper_triangle(input_x)

                x_in_np = np.random.random((1, 4, 32, 32)).astype(dtype)
                rst_np = _get_softmax_upper(x_in_np, dtype == 'float16')

                exe = fluid.Executor(fluid.CUDAPlace(0))
                fetches = exe.run(fluid.default_main_program(),
                                  feed={"x": x_in_np},
                                  fetch_list=[rst])
                np.testing.assert_allclose(fetches[0], rst_np, rtol=1e-05)

    def test_dygraph(self):
        for dtype in self.dtypes:
            with fluid.dygraph.guard(fluid.CUDAPlace(0)):
                x_in_np = np.random.random((1, 4, 32, 32)).astype(dtype)
                rst_np = _get_softmax_upper(x_in_np, dtype == 'float16')
                input_x = fluid.dygraph.to_variable(x_in_np)

                rst = incubate.softmax_mask_fuse_upper_triangle(input_x)
                np.testing.assert_allclose(rst, rst_np, rtol=1e-05)


if __name__ == '__main__':
    unittest.main()
