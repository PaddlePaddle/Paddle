#   Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest, is_custom_device

import paddle
from paddle.base import core

np.random.seed(0)
paddle.enable_static()


def max_with_index(x, dim=None, keepdim=False):
    """makeshift wrapper for the C++ op, extracted from compat.max"""
    vals, inds = paddle._C_ops.max_with_index(x, dim, keepdim, False)
    inds.stop_gradient = True
    return vals, inds


def min_with_index(x, dim=None, keepdim=False):
    """makeshift wrapper for the C++ op, extracted from compat.min"""
    vals, inds = paddle._C_ops.min_with_index(x, dim, keepdim, False)
    inds.stop_gradient = True
    return vals, inds


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device()),
    "core is not compiled with CUDA, skipping",
)
class TestMaxWithIndexBasic(OpTest):
    def setUp(self):
        self.set_op_input_attr()
        self.set_testing_op()
        self.set_data_type()
        self.set_input_shape()
        if self.is_int:
            inputs = np.random.randint(0, 255, self.input_shape).astype(
                self.dtype
            )
        else:
            inputs = np.random.rand(*self.input_shape).astype(self.dtype)

        self.prim_op_type = "prim"
        self.python_out_sig = ["values", "indices"]
        self.attrs = {"dim": self.dim, "keepdim": self.keepdim}

        gt_values = self.value_op(inputs, axis=self.dim, keepdims=self.keepdim)
        gt_indices = self.index_op(inputs, axis=self.dim, keepdims=self.keepdim)
        self.inputs = {
            'x': inputs,
        }
        self.outputs = {
            'values': gt_values,
            'indices': gt_indices,
        }

    def compute_grad(self):
        grad = np.zeros_like(self.inputs['x'], dtype=self.dtype)
        indices = (
            self.outputs['indices']
            if self.keepdim
            else np.expand_dims(self.outputs['indices'], axis=self.dim)
        )
        np.put_along_axis(grad, indices, 1, axis=self.dim)
        return grad

    def set_testing_op(self):
        self.op_type = "max_with_index"
        self.python_api = max_with_index
        self.public_python_api = max_with_index
        self.value_op = np.max
        self.index_op = np.argmax

    def set_data_type(self):
        self.dtype = np.float64
        self.is_int = False

    def set_input_shape(self):
        self.input_shape = [30, 257, 21]

    def set_op_input_attr(self):
        self.dim = 0
        self.keepdim = False

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        grad = self.compute_grad()
        self.check_grad(
            ['x'],
            'values',
            check_pir=True,
            user_defined_grads=[grad * (1.0 / grad.sum())],
        )


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device()),
    "core is not compiled with CUDA, skipping",
)
class TestMinWithIndexBasic(TestMaxWithIndexBasic):
    def set_testing_op(self):
        self.op_type = "min_with_index"
        self.python_api = min_with_index
        self.public_python_api = min_with_index
        self.value_op = np.min
        self.index_op = np.argmin


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device()),
    "core is not compiled with CUDA, skipping",
)
class TestMinWithIndexKeepDim(TestMinWithIndexBasic):
    def set_op_input_attr(self):
        self.dim = 1
        self.keepdim = True


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device()),
    "core is not compiled with CUDA, skipping",
)
class TestMaxWithIndexKeepDim(TestMaxWithIndexBasic):
    def set_op_input_attr(self):
        self.dim = 1
        self.keepdim = True


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device()),
    "core is not compiled with CUDA, skipping",
)
class TestMinWithIndexNegDim(TestMinWithIndexBasic):
    def set_op_input_attr(self):
        self.dim = -1
        self.keepdim = False


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device()),
    "core is not compiled with CUDA, skipping",
)
class TestMaxWithIndexNegDim(TestMaxWithIndexBasic):
    def set_op_input_attr(self):
        self.dim = 1
        self.keepdim = False


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device()),
    "core is not compiled with CUDA, skipping",
)
class TestMinWithIndexMoreTypeAndShape(TestMinWithIndexBasic):
    def set_op_input_attr(self):
        self.dim = 1
        self.keepdim = True

    def set_data_type(self):
        self.dtype = np.float32
        self.is_int = False

    def set_input_shape(self):
        self.input_shape = [10, 20, 16]


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device()),
    "core is not compiled with CUDA, skipping",
)
class TestMinWithIndexFP16(TestMinWithIndexBasic):
    def set_data_type(self):
        self.dtype = np.float16
        self.is_int = False


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device()),
    "core is not compiled with CUDA, skipping",
)
class TestMaxWithIndexU8(TestMaxWithIndexBasic):
    def set_data_type(self):
        self.dtype = np.uint8
        self.is_int = True

    @unittest.skipIf(
        True,
        "integral type does not need to check grad",
    )
    def test_check_grad(self):
        pass


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device()),
    "core is not compiled with CUDA, skipping",
)
class TestMaxWithIndexMoreTypeAndShape(TestMaxWithIndexBasic):
    def set_op_input_attr(self):
        self.dim = -1
        self.keepdim = False

    def set_data_type(self):
        self.dtype = np.uint8
        self.is_int = True

    def set_input_shape(self):
        self.input_shape = [4095]

    @unittest.skipIf(
        True,
        "integral type does not need to check grad",
    )
    def test_check_grad(self):
        pass


if __name__ == "__main__":
    unittest.main()
