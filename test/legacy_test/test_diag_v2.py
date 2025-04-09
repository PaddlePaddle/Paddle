#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest, convert_float_to_uint16

import paddle
from paddle import base, static
from paddle.base import core


class TestDiagV2Op(OpTest):
    def setUp(self):
        self.op_type = "diag_v2"
        self.python_api = paddle.diag
        self.prim_op_type = "comp"
        self.public_python_api = paddle.diag

        self.init_dtype()
        self.init_attrs()
        self.init_input_output()
        self.set_input_output()

    def init_dtype(self):
        self.dtype = np.float64

    def init_attrs(self):
        self.offset = 0
        self.padding_value = 0.0

    def init_input_output(self):
        self.x = np.random.rand(10, 10).astype(self.dtype)
        self.out = np.diag(self.x, self.offset)

    def set_input_output(self):
        self.attrs = {
            'offset': self.offset,
            'padding_value': self.padding_value,
        }

        self.inputs = {'X': self.x}
        self.outputs = {'Out': self.out}

    def test_check_output(self):
        paddle.enable_static()
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            self.check_output(check_pir=True)
        else:
            self.check_output(check_pir=True, check_prim_pir=True)

    def test_check_grad(self):
        paddle.enable_static()
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            self.check_grad(['X'], 'Out', check_pir=True)
        else:
            self.check_grad(['X'], 'Out', check_pir=True, check_prim_pir=True)


class TestDiagV2OpCase1(TestDiagV2Op):
    def init_attrs(self):
        super().init_attrs()
        self.offset = 1


class TestDiagV2OpCase2(TestDiagV2Op):
    def init_attrs(self):
        super().init_attrs()
        self.offset = -1


class TestDiagV2OpCase3(TestDiagV2Op):
    def init_input_output(self):
        self.x = np.random.randint(-10, 10, size=(10, 10)).astype(self.dtype)
        self.out = np.diag(self.x, self.offset)


class TestDiagV2OpCase4(TestDiagV2Op):
    def init_dtype(self):
        self.dtype = np.float32

    def init_attrs(self):
        super().init_attrs()
        self.padding_value = 2

    def init_input_output(self):
        self.x = np.random.rand(100).astype(self.dtype)
        n = self.x.size
        self.out = (
            self.padding_value * np.ones((n, n))
            + np.diag(self.x, self.offset)
            - np.diag(self.padding_value * np.ones(n))
        )


class TestDiagV2Error(unittest.TestCase):

    def test_errors(self):
        paddle.enable_static()
        main = static.Program()
        startup = static.Program()
        with static.program_guard(main, startup):

            def test_diag_v2_type():
                x = [1, 2, 3]
                output = paddle.diag(x)

            self.assertRaises(TypeError, test_diag_v2_type)

            x = paddle.static.data('data', [3, 3])
            self.assertRaises(TypeError, paddle.diag, x, offset=2.5)

            self.assertRaises(TypeError, paddle.diag, x, padding_value=[9])

            x = paddle.static.data('data2', [3, 3, 3])
            self.assertRaises(ValueError, paddle.diag, x)


class TestDiagV2API(unittest.TestCase):
    def setUp(self):
        self.input_np = np.random.random(size=(10, 10)).astype(np.float32)
        self.expected0 = np.diag(self.input_np)
        self.expected1 = np.diag(self.input_np, k=1)
        self.expected2 = np.diag(self.input_np, k=-1)

        self.input_np2 = np.random.rand(100)
        self.offset = 0
        self.padding_value = 8
        n = self.input_np2.size
        self.expected3 = (
            self.padding_value * np.ones((n, n))
            + np.diag(self.input_np2, self.offset)
            - np.diag(self.padding_value * np.ones(n))
        )

        self.input_np3 = np.random.randint(-10, 10, size=(100)).astype(np.int64)
        self.padding_value = 8.0
        n = self.input_np3.size
        self.expected4 = (
            self.padding_value * np.ones((n, n))
            + np.diag(self.input_np3, self.offset)
            - np.diag(self.padding_value * np.ones(n))
        )

        self.padding_value = -8
        self.expected5 = (
            self.padding_value * np.ones((n, n))
            + np.diag(self.input_np3, self.offset)
            - np.diag(self.padding_value * np.ones(n))
        )

        self.input_np4 = np.random.random(size=(2000, 2000)).astype(np.float32)
        self.expected6 = np.diag(self.input_np4)
        self.expected7 = np.diag(self.input_np4, k=1)
        self.expected8 = np.diag(self.input_np4, k=-1)

        self.input_np5 = np.random.random(size=(2000)).astype(np.float32)
        self.expected9 = np.diag(self.input_np5)
        self.expected10 = np.diag(self.input_np5, k=1)
        self.expected11 = np.diag(self.input_np5, k=-1)

        self.input_np6 = np.random.random(size=(2000, 1500)).astype(np.float32)
        self.expected12 = np.diag(self.input_np6, k=-1)

    def run_imperative(self):
        x = paddle.to_tensor(self.input_np)
        y = paddle.diag(x)
        np.testing.assert_allclose(y.numpy(), self.expected0, rtol=1e-05)

        y = paddle.diag(x, offset=1)
        np.testing.assert_allclose(y.numpy(), self.expected1, rtol=1e-05)

        y = paddle.diag(x, offset=-1)
        np.testing.assert_allclose(y.numpy(), self.expected2, rtol=1e-05)

        x = paddle.to_tensor(self.input_np2)
        y = paddle.diag(x, padding_value=8)
        np.testing.assert_allclose(y.numpy(), self.expected3, rtol=1e-05)

        x = paddle.to_tensor(self.input_np3)
        y = paddle.diag(x, padding_value=8.0)
        np.testing.assert_allclose(y.numpy(), self.expected4, rtol=1e-05)

        y = paddle.diag(x, padding_value=-8)
        np.testing.assert_allclose(y.numpy(), self.expected5, rtol=1e-05)

        x = paddle.to_tensor(self.input_np4)
        y = paddle.diag(x)
        np.testing.assert_allclose(y.numpy(), self.expected6, rtol=1e-05)

        y = paddle.diag(x, offset=1)
        np.testing.assert_allclose(y.numpy(), self.expected7, rtol=1e-05)

        y = paddle.diag(x, offset=-1)
        np.testing.assert_allclose(y.numpy(), self.expected8, rtol=1e-05)

        x = paddle.to_tensor(self.input_np5)
        y = paddle.diag(x)
        np.testing.assert_allclose(y.numpy(), self.expected9, rtol=1e-05)

        y = paddle.diag(x, offset=1)
        np.testing.assert_allclose(y.numpy(), self.expected10, rtol=1e-05)

        y = paddle.diag(x, offset=-1)
        np.testing.assert_allclose(y.numpy(), self.expected11, rtol=1e-05)

        x = paddle.to_tensor(self.input_np6)
        y = paddle.diag(x, offset=-1)
        np.testing.assert_allclose(y.numpy(), self.expected12, rtol=1e-05)

    def run_static(self, use_gpu=False):
        mp, sp = static.Program(), static.Program()
        with static.program_guard(mp, sp):
            x = paddle.static.data(
                name='input', shape=[10, 10], dtype='float32'
            )
            x2 = paddle.static.data(name='input2', shape=[100], dtype='float64')
            x3 = paddle.static.data(name='input3', shape=[100], dtype='int64')
            x4 = paddle.static.data(
                name='input4', shape=[2000, 2000], dtype='float32'
            )
            x5 = paddle.static.data(
                name='input5', shape=[2000], dtype='float32'
            )
            x6 = paddle.static.data(
                name='input6', shape=[2000, 1500], dtype='float32'
            )
            result0 = paddle.diag(x)
            result1 = paddle.diag(x, offset=1)
            result2 = paddle.diag(x, offset=-1)
            result4 = paddle.diag(x2, padding_value=8)
            result5 = paddle.diag(x3, padding_value=8.0)
            result6 = paddle.diag(x3, padding_value=-8)
            result7 = paddle.diag(x4)
            result8 = paddle.diag(x4, offset=1)
            result9 = paddle.diag(x4, offset=-1)
            result10 = paddle.diag(x5)
            result11 = paddle.diag(x5, offset=1)
            result12 = paddle.diag(x5, offset=-1)
            result13 = paddle.diag(x6, offset=-1)

        place = base.CUDAPlace(0) if use_gpu else base.CPUPlace()
        exe = static.Executor(place)
        exe.run(sp)
        [
            res0,
            res1,
            res2,
            res4,
            res5,
            res6,
            res7,
            res8,
            res9,
            res10,
            res11,
            res12,
            res13,
        ] = exe.run(
            mp,
            feed={
                "input": self.input_np,
                "input2": self.input_np2,
                'input3': self.input_np3,
                'input4': self.input_np4,
                'input5': self.input_np5,
                'input6': self.input_np6,
            },
            fetch_list=[
                result0,
                result1,
                result2,
                result4,
                result5,
                result6,
                result7,
                result8,
                result9,
                result10,
                result11,
                result12,
                result13,
            ],
        )

        np.testing.assert_allclose(res0, self.expected0, rtol=1e-05)
        np.testing.assert_allclose(res1, self.expected1, rtol=1e-05)
        np.testing.assert_allclose(res2, self.expected2, rtol=1e-05)
        np.testing.assert_allclose(res4, self.expected3, rtol=1e-05)
        np.testing.assert_allclose(res5, self.expected4, rtol=1e-05)
        np.testing.assert_allclose(res6, self.expected5, rtol=1e-05)
        np.testing.assert_allclose(res7, self.expected6, rtol=1e-05)
        np.testing.assert_allclose(res8, self.expected7, rtol=1e-05)
        np.testing.assert_allclose(res9, self.expected8, rtol=1e-05)
        np.testing.assert_allclose(res10, self.expected9, rtol=1e-05)
        np.testing.assert_allclose(res11, self.expected10, rtol=1e-05)
        np.testing.assert_allclose(res12, self.expected11, rtol=1e-05)
        np.testing.assert_allclose(res13, self.expected12, rtol=1e-05)

    def test_cpu(self):
        paddle.disable_static(place=paddle.base.CPUPlace())
        self.run_imperative()

        paddle.enable_static()
        self.run_static()

    def test_gpu(self):
        if not base.core.is_compiled_with_cuda():
            return

        paddle.disable_static(place=paddle.base.CUDAPlace(0))
        self.run_imperative()
        paddle.enable_static()
        self.run_static(use_gpu=True)


class TestDiagV2FP16OP(TestDiagV2Op):
    def init_dtype(self):
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestDiagV2BF16OP(OpTest):
    def setUp(self):
        self.op_type = "diag_v2"
        self.python_api = paddle.diag
        self.prim_op_type = "comp"
        self.public_python_api = paddle.diag
        self.dtype = np.uint16
        x = np.random.rand(10, 10).astype(np.float32)
        offset = 0
        padding_value = 0.0
        out = np.diag(x, offset)

        self.inputs = {'X': convert_float_to_uint16(x)}
        self.attrs = {
            'offset': offset,
            'padding_value': padding_value,
        }
        self.outputs = {'Out': convert_float_to_uint16(out)}

    def test_check_output(self):
        paddle.enable_static()
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, check_pir=True, check_prim_pir=True)

    def test_check_grad(self):
        paddle.enable_static()
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place, ['X'], 'Out', check_pir=True, check_prim_pir=True
        )


@unittest.skipIf(
    core.is_compiled_with_xpu(),
    "xpu does not support complex64",
)
class TestDiagV2Complex64OP(TestDiagV2Op):
    def init_dtype(self):
        self.dtype = np.complex64

    def init_input_output(self):
        self.x = (
            np.random.randint(-10, 10, size=(10, 10))
            + 1j * np.random.randint(-10, 10, size=(10, 10))
        ).astype(self.dtype)
        self.out = np.diag(self.x, self.offset)


@unittest.skipIf(
    core.is_compiled_with_xpu(),
    "xpu does not support complex128",
)
class TestDiagV2Complex128OP(TestDiagV2Op):
    def init_dtype(self):
        self.dtype = np.complex128

    def init_config(self):
        self.x = (
            np.random.randint(-10, 10, size=(10, 10))
            + 1j * np.random.randint(-10, 10, size=(10, 10))
        ).astype(self.dtype)
        self.out = np.diag(self.x, self.offset)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
