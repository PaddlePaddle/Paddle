# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import utils

import paddle


class TestOpsBase(unittest.TestCase):
    """
    Test Pir API + @to_static + CINN.
    """

    def setUp(self):
        paddle.seed(2022)
        self.prepare_data()
        self.prepare_info()

    def prepare_info(self):
        self.fn = None
        self.expected_jit_kernel_number = 1
        self.expected_jit_kernel_structure = {utils.JIT_KERNEL_NAME: 1}

    def prepare_data(self):
        self.shape = [64, 128]
        self.axis = -1
        self.x = paddle.randn(self.shape, dtype="float32")
        self.x.stop_gradient = False
        self.y = paddle.randn(self.shape, dtype="float32")
        self.y.stop_gradient = False

    def check_eval(self):
        static_fn = utils.apply_to_static(self.fn, use_cinn=True)
        cinn_out = static_fn(self.x, self.y)
        dy_out = self.fn(self.x, self.y)
        np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-8)

        utils.check_jit_kernel_number(
            static_fn, self.expected_jit_kernel_number
        )
        utils.check_jit_kernel_structure(
            static_fn, self.expected_jit_kernel_structure
        )


class TestAddOp(TestOpsBase):
    def prepare_info(self):
        self.fn = paddle.add
        self.expected_jit_kernel_number = 1
        self.expected_jit_kernel_structure = {utils.JIT_KERNEL_NAME: 1}

    def test_eval(self):
        self.check_eval()


class TestIsCloseOp(TestOpsBase):
    def prepare_info(self):
        self.fn = paddle.sin
        self.expected_jit_kernel_number = 1
        self.expected_jit_kernel_structure = {utils.JIT_KERNEL_NAME: 1}

    def test_eval(self):
        self.check_eval()


class TestTransposeOp(TestOpsBase):
    def prepare_info(self):
        self.fn = paddle.transpose
        self.expected_jit_kernel_number = 1
        self.expected_jit_kernel_structure = {utils.JIT_KERNEL_NAME: 1}

    def prepare_data(self):
        self.shape = [16, 3, 10, 32]
        self.perm = [1, 0, 2]
        self.x = paddle.randn(self.shape, dtype="float32")
        self.x.stop_gradient = False

    def test_eval(self):
        static_fn = utils.apply_to_static(self.fn, use_cinn=True)
        cinn_out = static_fn(self.x, self.perm)
        dy_out = self.fn(self.x, self.perm)
        np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-8)

        utils.check_jit_kernel_number(
            static_fn, self.expected_jit_kernel_number
        )
        utils.check_jit_kernel_structure(
            static_fn, self.expected_jit_kernel_structure
        )


if __name__ == '__main__':
    unittest.main()
