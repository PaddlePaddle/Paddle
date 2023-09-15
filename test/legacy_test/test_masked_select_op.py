# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle import base
from paddle.base import core


def np_masked_fill(x, mask, value):
    if not np.isscalar(value):
        value = value[0]

    x, mask = np.broadcast_arrays(x, mask)
    result = np.copy(x)
    for idx, m in np.ndenumerate(mask):
        if m:
            result[idx] = value
    return result


paddle.enable_static()


class TestMaskedFillOp(unittest.TestCase):
    def setUp(self):
        self.init()

        self.x_np = np.random.random(self.x_shape).astype("float32")
        self.mask_np = np.array(
            np.random.randint(2, size=self.mask_shape), dtype="bool"
        )

        self.value_np = np.random.randn(1).astype("float32")
        self.out_np = np_masked_fill(self.x_np, self.mask_np, self.value_np)

    def init(self):
        self.x_shape = (50, 3)
        self.mask_shape = self.x_shape

    def test_static_graph(self):
        paddle.enable_static()
        startup_program = base.Program()
        train_program = base.Program()
        with base.program_guard(startup_program, train_program):
            x = paddle.static.data(
                name='x', dtype='float32', shape=self.x_shape
            )
            mask = paddle.static.data(
                name='mask', dtype='bool', shape=self.mask_shape
            )
            value = paddle.static.data(
                name='value', dtype='float32', shape=self.value_np.shape
            )
            out = paddle.masked_fill(x, mask, value)

            place = (
                base.CUDAPlace(0)
                if core.is_compiled_with_cuda()
                else base.CPUPlace()
            )
            exe = base.Executor(place)
            res = exe.run(
                base.default_main_program(),
                feed={
                    'x': self.x_np,
                    'mask': self.mask_np,
                    'value': self.value_np,
                },
                fetch_list=[out],
            )
            np.testing.assert_allclose(
                res[0], self.out_np, atol=1e-5, rtol=1e-5
            )
            paddle.disable_static()

    def test_dygraph(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x_np, dtype='float32')
        mask = paddle.to_tensor(self.mask_np).astype('bool')
        value = paddle.to_tensor(self.value_np, dtype='float32')
        result = paddle.masked_fill(x, mask, value)
        np.testing.assert_allclose(self.out_np, result.numpy(), rtol=1e-05)

        paddle.enable_static()


class TestMaskedFillOp1(TestMaskedFillOp):
    def init(self):
        self.x_shape = (6, 8, 9, 18)
        self.mask_shape = self.x_shape


class TestMaskedFillOp2(TestMaskedFillOp):
    def init(self):
        self.x_shape = (168,)
        self.mask_shape = self.x_shape


class TestMaskedFillOpBroadcast(TestMaskedFillOp):
    def init(self):
        self.x_shape = (3, 40)
        self.mask_shape = (3, 1)


class TestMaskedFillOpBroadcast2(TestMaskedFillOp):
    def init(self):
        self.x_shape = (300, 1)
        self.mask_shape = (300, 40)


class TestMaskedFillOpBroadcast3(TestMaskedFillOp):
    def init(self):
        self.x_shape = (120,)
        self.mask_shape = (300, 120)


class TestMaskedFillOpBroadcast4(TestMaskedFillOp):
    def init(self):
        self.x_shape = (300, 40)
        self.mask_shape = (40,)


# class TestMaskedFillFP16Op(OpTest):
#     def setUp(self):
#         self.init()
#         self.op_type = "masked_fill"
#         self.dtype = np.float16
#         self.python_api = paddle.masked_fill
#         x = np.random.random(self.shape).astype("float16")
#         mask = np.array(np.random.randint(2, size=self.shape, dtype=bool))
#         value = np.array(np.random.random(5).astype("float16"))
#         out = np_masked_fill(x, mask, value)
#         self.inputs = {'X': x, 'Mask': mask}
#         self.outputs = {'Y': out}

#     def test_check_output(self):
#         self.check_output()

#     def test_check_grad(self):
#         self.check_grad(['X'], 'Y')

#     def init(self):
#         self.shape = (50, 3)


# class TestMaskedFillFP16Op1(TestMaskedFillFP16Op):
#     def init(self):
#         self.shape = (6, 8, 9, 18)


# class TestMaskedFillFP16Op2(TestMaskedFillFP16Op):
#     def init(self):
#         self.shape = (168,)


# @unittest.skipIf(
#     not core.is_compiled_with_cuda()
#     or not core.is_bfloat16_supported(core.CUDAPlace(0)),
#     "core is not compiled with CUDA or not support bfloat16",
# )
# class TestMaskedFillBF16Op(OpTest):
#     def setUp(self):
#         self.init()
#         self.op_type = "masked_fill"
#         self.dtype = np.uint16
#         self.python_api = paddle.masked_fill
#         x = np.random.random(self.shape).astype("float32")
#         mask = np.array(np.random.randint(2, size=self.shape, dtype=bool))
#         out = np_masked_fill(x, mask)
#         self.inputs = {'X': convert_float_to_uint16(x), 'Mask': mask}
#         self.outputs = {'Y': convert_float_to_uint16(out)}

#     def test_check_output(self):
#         self.check_output_with_place(core.CUDAPlace(0))

#     def test_check_grad(self):
#         self.check_grad_with_place(core.CUDAPlace(0), ['X'], 'Y')

#     def init(self):
#         self.shape = (50, 3)


# class TestMaskedFillBF16Op1(TestMaskedFillBF16Op):
#     def init(self):
#         self.shape = (6, 8, 9, 2)


# class TestMaskedFillBF16Op2(TestMaskedFillBF16Op):
#     def init(self):
#         self.shape = (168,)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
