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


class TestMaskedFillAPI(unittest.TestCase):
    def setUp(self):
        self.init()

        self.x_np = np.random.random(self.x_shape).astype(self.dtype)
        self.mask_np = np.array(
            np.random.randint(2, size=self.mask_shape), dtype="bool"
        )

        self.value_np = np.random.randn(1).astype(self.dtype)
        self.out_np = np_masked_fill(self.x_np, self.mask_np, self.value_np)

    def init(self):
        self.x_shape = (50, 3)
        self.mask_shape = self.x_shape
        self.dtype = "float32"

    def test_static_graph(self):
        paddle.enable_static()
        startup_program = base.Program()
        train_program = base.Program()
        with base.program_guard(startup_program, train_program):
            x = paddle.static.data(
                name='x', dtype=self.dtype, shape=self.x_shape
            )
            mask = paddle.static.data(
                name='mask', dtype='bool', shape=self.mask_shape
            )
            value = paddle.static.data(
                name='value', dtype=self.dtype, shape=self.value_np.shape
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
        x = paddle.to_tensor(self.x_np, dtype=self.dtype)
        mask = paddle.to_tensor(self.mask_np).astype('bool')
        value = paddle.to_tensor(self.value_np, dtype=self.dtype)
        result = paddle.masked_fill(x, mask, value)
        np.testing.assert_allclose(self.out_np, result.numpy(), rtol=1e-05)

        paddle.enable_static()


class TestMaskedFillAPI1(TestMaskedFillAPI):
    def init(self):
        self.x_shape = (6, 8, 9, 18)
        self.mask_shape = self.x_shape
        self.dtype = "float32"


class TestMaskedFillAPI2(TestMaskedFillAPI):
    def init(self):
        self.x_shape = (168,)
        self.mask_shape = self.x_shape
        self.dtype = "float32"


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestMaskedFillFP16API1(TestMaskedFillAPI):
    def init(self):
        self.x_shape = (6, 8, 9, 18)
        self.mask_shape = self.x_shape
        self.dtype = "float16"


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestMaskedFillFP16API2(TestMaskedFillAPI):
    def init(self):
        self.x_shape = (168,)
        self.mask_shape = self.x_shape
        self.dtype = "float16"


class TestMaskedFillAPIBroadcast(TestMaskedFillAPI):
    def init(self):
        self.x_shape = (3, 40)
        self.mask_shape = (3, 1)
        self.dtype = "float32"


class TestMaskedFillAPIBroadcast2(TestMaskedFillAPI):
    def init(self):
        self.x_shape = (3, 3)
        self.mask_shape = (1, 3)
        self.dtype = "float32"


class TestMaskedFillAPIBroadcast3(TestMaskedFillAPI):
    def init(self):
        self.x_shape = (120,)
        self.mask_shape = (300, 120)
        self.dtype = "float32"


class TestMaskedFillAPIBroadcast4(TestMaskedFillAPI):
    def init(self):
        self.x_shape = (300, 40)
        self.mask_shape = (40,)
        self.dtype = "float32"


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestMaskedFillFP16APIBroadcast(TestMaskedFillAPI):
    def init(self):
        self.x_shape = (3, 40)
        self.mask_shape = (3, 1)
        self.dtype = "float16"


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestMaskedFillFP16APIBroadcast2(TestMaskedFillAPI):
    def init(self):
        self.x_shape = (300, 1)
        self.mask_shape = (300, 40)
        self.dtype = "float16"


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support bfloat16",
)
class TestMaskedFillBF16(TestMaskedFillAPI):
    def init(self):
        self.x_shape = (300, 1)
        self.mask_shape = (300, 1)
        self.dtype = "uint16"


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support bfloat16",
)
class TestMaskedFillBF16APIBroadcast2(TestMaskedFillAPI):
    def init(self):
        self.x_shape = (300, 1)
        self.mask_shape = (300, 3)
        self.dtype = "uint16"


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
