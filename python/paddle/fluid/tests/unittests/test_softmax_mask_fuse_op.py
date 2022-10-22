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


def _get_softmax(x, mask, fp16=True):
    masked_x = (x + mask).astype("float32")
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
        self.op_type = "fused_softmax_mask"
        x = np.random.random((1, 1, 8, 32))
        mask = np.random.randint(0, 2, (1, 1, 8, 32))
        mask_input = np.where(mask == 1, -10000.0, mask)
        self.inputs = {'X': x, 'Mask': mask_input}
        rst = _get_softmax(x, mask_input)
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
class TestSoftmaxMaskFuseOp0(OpTest):

    def setUp(self):
        self.op_type = "fused_softmax_mask"
        x = np.random.random((1, 1, 8, 32)).astype("float16")
        mask = np.random.randint(0, 2, (1, 1, 8, 32)).astype("float16")
        mask_input = np.where(mask == 1, -10000.0, mask)
        self.inputs = {'X': x, 'Mask': mask_input}
        rst = _get_softmax(x, mask_input)
        self.outputs = {'Out': rst}

    def test_check_output(self):
        self.check_output_with_place(core.CUDAPlace(0))

    def test_check_grad(self):
        self.check_grad_with_place(core.CUDAPlace(0), ["X"], "Out")


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestDropoutBiasFuseOp3(unittest.TestCase):

    def test_static_result(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            input_x = fluid.data(name="x", shape=[1, 1, 8, 32], dtype="float32")
            input_mask = fluid.data(name="mask",
                                    shape=[1, 1, 8, 32],
                                    dtype="float32")
            rst = incubate.softmax_mask_fuse(input_x, input_mask)

            x_in_np = np.random.random((1, 1, 8, 32)).astype("float32")
            mask = np.random.randint(0, 2, (1, 1, 8, 32)).astype("float32")
            mask_in_np = np.where(mask == 1, -10000.0, mask)
            rst_np = _get_softmax(x_in_np, mask_in_np, False)

            exe = fluid.Executor(fluid.CUDAPlace(0))
            fetches = exe.run(fluid.default_main_program(),
                              feed={
                                  "x": x_in_np,
                                  "mask": mask_in_np
                              },
                              fetch_list=[rst])
            np.testing.assert_allclose(fetches[0], rst_np, rtol=1e-05)

    def test_dygraph(self):
        with fluid.dygraph.guard(fluid.CUDAPlace(0)):
            x_in_np = np.random.random((1, 1, 8, 32)).astype("float32")
            mask = np.random.randint(0, 2, (1, 1, 8, 32)).astype("float32")
            mask_in_np = np.where(mask == 1, -10000.0, mask)
            rst_np = _get_softmax(x_in_np, mask_in_np, False)
            input_x = fluid.dygraph.to_variable(x_in_np)
            input_mask = fluid.dygraph.to_variable(mask_in_np)

            rst = incubate.softmax_mask_fuse(input_x, input_mask)
            np.testing.assert_allclose(rst, rst_np, rtol=1e-05)


if __name__ == '__main__':
    unittest.main()
