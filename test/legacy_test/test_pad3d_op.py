#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest

import numpy as np
from op_test import OpTest, convert_float_to_uint16

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.base import (
    Executor,
    core,
)


class TestPad3dOp(OpTest):
    def setUp(self):
        paddle.enable_static()
        self.value = 0.0
        self.initTestCase()
        self.dtype = self.get_dtype()
        self.op_type = "pad3d"
        self.python_api = paddle.nn.functional.pad
        self.inputs = {
            'X': (
                np.random.uniform(-1.0, 1.0, self.shape).astype("float32")
                if self.dtype == np.uint16
                else (
                    (
                        np.random.uniform(-1.0, 1.0, self.shape)
                        + 1j * np.random.uniform(-1.0, 1.0, self.shape)
                    ).astype(self.dtype)
                    if self.dtype == np.complex64 or self.dtype == np.complex128
                    else np.random.uniform(-1.0, 1.0, self.shape).astype(
                        self.dtype
                    )
                )
            )
        }
        self.attrs = {}
        if self.variable_paddings:
            self.attrs['paddings'] = []
            self.inputs['Paddings'] = (
                np.array(self.paddings).flatten().astype("int32")
            )
        else:
            self.attrs['paddings'] = (
                np.array(self.paddings).flatten().astype("int32")
            )
        self.attrs['value'] = self.value
        self.attrs['mode'] = self.mode
        self.attrs['data_format'] = self.data_format
        if self.data_format == "NCDHW":
            paddings = [
                (0, 0),
                (0, 0),
                (self.paddings[4], self.paddings[5]),
                (self.paddings[2], self.paddings[3]),
                (self.paddings[0], self.paddings[1]),
            ]
        else:
            paddings = [
                (0, 0),
                (self.paddings[4], self.paddings[5]),
                (self.paddings[2], self.paddings[3]),
                (self.paddings[0], self.paddings[1]),
                (0, 0),
            ]
        if self.mode == "constant":
            out = np.pad(
                self.inputs['X'],
                paddings,
                mode=self.mode,
                constant_values=self.value,
            )
        elif self.mode == "reflect":
            out = np.pad(self.inputs['X'], paddings, mode=self.mode)
        elif self.mode == "replicate":
            out = np.pad(self.inputs['X'], paddings, mode="edge")
        elif self.mode == "circular":
            out = np.pad(self.inputs['X'], paddings, mode="wrap")
        self.outputs = {'Out': out}

        if self.dtype == np.uint16:
            self.inputs['X'] = convert_float_to_uint16(self.inputs['X'])
            self.outputs['Out'] = convert_float_to_uint16(self.outputs['Out'])

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out', check_pir=True)

    def get_dtype(self):
        return np.float64

    def initTestCase(self):
        self.shape = (2, 3, 4, 5, 6)
        self.paddings = [0, 0, 0, 0, 0, 0]
        self.mode = "constant"
        self.data_format = "NCDHW"
        self.pad_value = 0.0
        self.variable_paddings = False


class TestCase1(TestPad3dOp):
    def initTestCase(self):
        self.shape = (2, 3, 4, 5, 6)
        self.paddings = [0, 1, 2, 3, 4, 5]
        self.mode = "constant"
        self.data_format = "NCDHW"
        self.value = 1.0
        self.variable_paddings = False


class TestCase2(TestPad3dOp):
    def initTestCase(self):
        self.shape = (2, 3, 4, 5, 6)
        self.paddings = [1, 1, 1, 1, 1, 1]
        self.mode = "constant"
        self.data_format = "NDHWC"
        self.value = 1.0
        self.variable_paddings = False


class TestCase3(TestPad3dOp):
    def initTestCase(self):
        self.shape = (2, 3, 4, 5, 6)
        self.paddings = [0, 1, 1, 0, 2, 3]
        self.mode = "reflect"
        self.data_format = "NCDHW"
        self.variable_paddings = False


class TestCase4(TestPad3dOp):
    def initTestCase(self):
        self.shape = (4, 4, 4, 4, 4)
        self.paddings = [0, 1, 2, 1, 2, 3]
        self.mode = "reflect"
        self.data_format = "NDHWC"
        self.variable_paddings = False


class TestCase5(TestPad3dOp):
    def initTestCase(self):
        self.shape = (2, 3, 4, 5, 6)
        self.paddings = [0, 1, 2, 3, 2, 1]
        self.mode = "replicate"
        self.data_format = "NCDHW"
        self.variable_paddings = False


class TestCase6(TestPad3dOp):
    def initTestCase(self):
        self.shape = (4, 4, 4, 4, 4)
        self.paddings = [5, 4, 2, 1, 2, 3]
        self.mode = "replicate"
        self.data_format = "NDHWC"
        self.variable_paddings = False


class TestCase7(TestPad3dOp):
    def initTestCase(self):
        self.shape = (2, 3, 4, 5, 6)
        self.paddings = [0, 1, 2, 3, 2, 1]
        self.mode = "circular"
        self.data_format = "NCDHW"
        self.variable_paddings = False


class TestCase8(TestPad3dOp):
    def initTestCase(self):
        self.shape = (4, 4, 4, 4, 4)
        self.paddings = [0, 1, 2, 1, 2, 3]
        self.mode = "circular"
        self.data_format = "NDHWC"
        self.variable_paddings = False


class TestCase9(TestPad3dOp):
    def initTestCase(self):
        self.shape = (2, 3, 4, 5, 6)
        self.paddings = [0, 1, 2, 3, 4, 5]
        self.mode = "constant"
        self.data_format = "NCDHW"
        self.value = 1.0
        self.variable_paddings = True

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=False)


class TestCase10(TestPad3dOp):
    def initTestCase(self):
        self.shape = (2, 3, 4, 5, 6)
        self.paddings = [0, 1, 2, 3, 4, 5]
        self.mode = "constant"
        self.data_format = "NDHWC"
        self.value = 1.0
        self.variable_paddings = True

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=False)


# ----------------Pad3d Fp16----------------


def create_test_fp16(parent):
    @unittest.skipIf(
        not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
    )
    class TestPad3dFp16(parent):
        def get_dtype(self):
            return np.float16

        def test_check_output(self):
            self.check_output(
                atol=1e-3,
                check_pir=True,
                check_symbol_infer=(not self.variable_paddings),
            )

        def test_check_grad_normal(self):
            self.check_grad(
                ['X'], 'Out', max_relative_error=1.5e-3, check_pir=True
            )

    cls_name = "{}_{}".format(parent.__name__, "FP16OP")
    TestPad3dFp16.__name__ = cls_name
    globals()[cls_name] = TestPad3dFp16


create_test_fp16(TestCase1)
create_test_fp16(TestCase2)
create_test_fp16(TestCase3)
create_test_fp16(TestCase4)
create_test_fp16(TestCase5)
create_test_fp16(TestCase6)
create_test_fp16(TestCase7)
create_test_fp16(TestCase8)
create_test_fp16(TestCase9)
create_test_fp16(TestCase10)


# ----------------Pad3d Bf16----------------


def create_test_bf16(parent):
    @unittest.skipIf(
        not core.is_compiled_with_cuda()
        or not core.is_bfloat16_supported(core.CUDAPlace(0)),
        "core is not compiled with CUDA and do not support bfloat16",
    )
    class TestPad3dBf16(parent):
        def get_dtype(self):
            return np.uint16

        def test_check_output(self):
            place = core.CUDAPlace(0)
            self.check_output_with_place(
                place,
                atol=1e-2,
                check_pir=True,
                check_symbol_infer=(not self.variable_paddings),
            )

        def test_check_grad_normal(self):
            place = core.CUDAPlace(0)
            self.check_grad_with_place(
                place, ['X'], 'Out', max_relative_error=1e-2, check_pir=True
            )

    cls_name = "{}_{}".format(parent.__name__, "BF16OP")
    TestPad3dBf16.__name__ = cls_name
    globals()[cls_name] = TestPad3dBf16


create_test_bf16(TestCase1)
create_test_bf16(TestCase2)
create_test_bf16(TestCase3)
create_test_bf16(TestCase4)
create_test_bf16(TestCase5)
create_test_bf16(TestCase6)
create_test_bf16(TestCase7)
create_test_bf16(TestCase8)
create_test_bf16(TestCase9)
create_test_bf16(TestCase10)


# ----------------Pad3d complex64----------------
def create_test_complex64(parent):
    @unittest.skipIf(
        not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
    )
    class TestPad3dComplex64(parent):
        def get_dtype(self):
            return np.complex64

        def test_check_output(self):
            self.check_output(
                atol=1e-3,
                check_pir=True,
                check_symbol_infer=(not self.variable_paddings),
            )

        def test_check_grad_normal(self):
            self.check_grad(
                ['X'], 'Out', max_relative_error=1.5e-3, check_pir=True
            )

    cls_name = "{}_{}".format(parent.__name__, "Complex64OP")
    TestPad3dComplex64.__name__ = cls_name  # 重新修改TestPad3dFp16的类名
    globals()[cls_name] = TestPad3dComplex64


create_test_complex64(TestCase1)
create_test_complex64(TestCase2)
create_test_complex64(TestCase3)
create_test_complex64(TestCase4)
create_test_complex64(TestCase5)
create_test_complex64(TestCase6)
create_test_complex64(TestCase7)
create_test_complex64(TestCase8)
create_test_complex64(TestCase9)
create_test_complex64(TestCase10)


# ----------------Pad3d complex128----------------


def create_test_complex128(parent):
    @unittest.skipIf(
        not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
    )
    class TestPad3dComplex128(parent):
        def get_dtype(self):
            return np.complex128

        def test_check_output(self):
            self.check_output(
                atol=1e-3,
                check_pir=True,
                check_symbol_infer=(not self.variable_paddings),
            )

        def test_check_grad_normal(self):
            self.check_grad(
                ['X'], 'Out', max_relative_error=1.5e-3, check_pir=True
            )

    cls_name = "{}_{}".format(parent.__name__, "Complex128OP")
    TestPad3dComplex128.__name__ = cls_name  # 重新修改TestPad3dFp16的类名
    globals()[cls_name] = TestPad3dComplex128


create_test_complex128(TestCase1)
create_test_complex128(TestCase2)
create_test_complex128(TestCase3)
create_test_complex128(TestCase4)
create_test_complex128(TestCase5)
create_test_complex128(TestCase6)
create_test_complex128(TestCase7)
create_test_complex128(TestCase8)
create_test_complex128(TestCase9)
create_test_complex128(TestCase10)


class TestPadAPI(unittest.TestCase):
    def setUp(self):
        self.init_dtype()
        self.places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.places.append(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def init_dtype(self):
        self.dtype = np.float32

    def check_static_result_1(self, place):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            input_shape = (1, 2, 3, 4, 5)
            pad = [1, 2, 1, 1, 3, 4]
            mode = "constant"
            value = 100
            input_data = np.random.rand(*input_shape).astype(self.dtype)
            if self.dtype == np.complex64 or self.dtype == np.complex128:
                input_data = (
                    np.random.rand(*input_shape)
                    + 1j * np.random.rand(*input_shape)
                ).astype(self.dtype)
            x = paddle.static.data(
                name="x", shape=input_shape, dtype=self.dtype
            )
            result = F.pad(
                x=x, pad=pad, value=value, mode=mode, data_format="NCDHW"
            )
            exe = Executor(place)
            fetches = exe.run(
                paddle.static.default_main_program(),
                feed={"x": input_data},
                fetch_list=[result],
            )

            np_out = self._get_numpy_out(input_data, pad, mode, value)
            np.testing.assert_allclose(fetches[0], np_out, rtol=1e-05)

    def check_static_result_2(self, place):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            input_shape = (2, 3, 4, 5, 6)
            pad = [1, 2, 1, 1, 1, 2]
            mode = "reflect"
            input_data = np.random.rand(*input_shape).astype(self.dtype)
            if self.dtype == np.complex64 or self.dtype == np.complex128:
                input_data = (
                    np.random.rand(*input_shape)
                    + 1j * np.random.rand(*input_shape)
                ).astype(self.dtype)
            x = paddle.static.data(
                name="x", shape=input_shape, dtype=self.dtype
            )
            result1 = F.pad(x=x, pad=pad, mode=mode, data_format="NCDHW")
            result2 = F.pad(x=x, pad=pad, mode=mode, data_format="NDHWC")
            result3 = F.pad(x=x, pad=pad, mode=mode)
            exe = Executor(place)
            fetches = exe.run(
                paddle.static.default_main_program(),
                feed={"x": input_data},
                fetch_list=[result1, result2, result3],
            )

            np_out1 = self._get_numpy_out(
                input_data, pad, mode, data_format="NCDHW"
            )
            np_out2 = self._get_numpy_out(
                input_data, pad, mode, data_format="NDHWC"
            )
            np.testing.assert_allclose(fetches[0], np_out1, rtol=1e-05)
            np.testing.assert_allclose(fetches[1], np_out2, rtol=1e-05)
            np.testing.assert_allclose(fetches[2], np_out1, rtol=1e-05)

    def check_static_result_3(self, place):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            input_shape = (2, 3, 4, 5, 6)
            pad = [1, 2, 1, 1, 3, 4]
            mode = "replicate"
            input_data = np.random.rand(*input_shape).astype(self.dtype)
            if self.dtype == np.complex64 or self.dtype == np.complex128:
                input_data = (
                    np.random.rand(*input_shape)
                    + 1j * np.random.rand(*input_shape)
                ).astype(self.dtype)
            x = paddle.static.data(
                name="x", shape=input_shape, dtype=self.dtype
            )
            result1 = F.pad(x=x, pad=pad, mode=mode, data_format="NCDHW")
            result2 = F.pad(x=x, pad=pad, mode=mode, data_format="NDHWC")
            result3 = F.pad(x=x, pad=pad, mode=mode)
            exe = Executor(place)
            fetches = exe.run(
                paddle.static.default_main_program(),
                feed={"x": input_data},
                fetch_list=[result1, result2, result3],
            )

            np_out1 = self._get_numpy_out(
                input_data, pad, mode, data_format="NCDHW"
            )
            np_out2 = self._get_numpy_out(
                input_data, pad, mode, data_format="NDHWC"
            )
            np.testing.assert_allclose(fetches[0], np_out1, rtol=1e-05)
            np.testing.assert_allclose(fetches[1], np_out2, rtol=1e-05)
            np.testing.assert_allclose(fetches[2], np_out1, rtol=1e-05)

    def check_static_result_4(self, place):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            input_shape = (2, 3, 4, 5, 6)
            pad = [1, 2, 1, 1, 3, 4]
            mode = "circular"
            input_data = np.random.rand(*input_shape).astype(self.dtype)
            if self.dtype == np.complex64 or self.dtype == np.complex128:
                input_data = (
                    np.random.rand(*input_shape)
                    + 1j * np.random.rand(*input_shape)
                ).astype(self.dtype)
            x = paddle.static.data(
                name="x", shape=input_shape, dtype=self.dtype
            )
            result1 = F.pad(x=x, pad=pad, mode=mode, data_format="NCDHW")
            result2 = F.pad(x=x, pad=pad, mode=mode, data_format="NDHWC")
            result3 = F.pad(x=x, pad=pad, mode=mode)
            exe = Executor(place)
            fetches = exe.run(
                paddle.static.default_main_program(),
                feed={"x": input_data},
                fetch_list=[result1, result2, result3],
            )

            np_out1 = self._get_numpy_out(
                input_data, pad, mode, data_format="NCDHW"
            )
            np_out2 = self._get_numpy_out(
                input_data, pad, mode, data_format="NDHWC"
            )
            np.testing.assert_allclose(fetches[0], np_out1, rtol=1e-05)
            np.testing.assert_allclose(fetches[1], np_out2, rtol=1e-05)
            np.testing.assert_allclose(fetches[2], np_out1, rtol=1e-05)

    def _get_numpy_out(
        self, input_data, pad, mode, value=0, data_format="NCDHW"
    ):
        if mode == "constant" and len(pad) == len(input_data.shape) * 2:
            pad = np.reshape(pad, (-1, 2)).tolist()
        elif data_format == "NCDHW":
            pad = [
                (0, 0),
                (0, 0),
                (pad[4], pad[5]),
                (pad[2], pad[3]),
                (pad[0], pad[1]),
            ]
        elif data_format == "NDHWC":
            pad = [
                (0, 0),
                (pad[4], pad[5]),
                (pad[2], pad[3]),
                (pad[0], pad[1]),
                (0, 0),
            ]
        elif data_format == "NCHW":
            pad = [
                (0, 0),
                (0, 0),
                (pad[2], pad[3]),
                (pad[0], pad[1]),
            ]
        elif data_format == "NHWC":
            pad = [
                (0, 0),
                (pad[2], pad[3]),
                (pad[0], pad[1]),
                (0, 0),
            ]
        elif data_format == "NCL":
            pad = [
                (0, 0),
                (0, 0),
                (pad[0], pad[1]),
            ]
        elif data_format == "NLC":
            pad = [
                (0, 0),
                (pad[0], pad[1]),
                (0, 0),
            ]

        if mode == "constant":
            out = np.pad(input_data, pad, mode=mode, constant_values=value)
        elif mode == "reflect":
            out = np.pad(input_data, pad, mode=mode)
        elif mode == "replicate":
            out = np.pad(input_data, pad, mode="edge")
        elif mode == "circular":
            out = np.pad(input_data, pad, mode="wrap")

        return out

    def test_static(self):
        for place in self.places:
            self.check_static_result_1(place=place)
            self.check_static_result_2(place=place)
            self.check_static_result_3(place=place)
            self.check_static_result_4(place=place)

    def test_dygraph_1(self):
        paddle.disable_static()
        input_shape = (1, 2, 3, 4, 5)
        pad = [1, 2, 1, 1, 3, 4]
        pad_3 = [1, 2, 1, 1, 3, 4, 5, 6, 7, 8]
        mode = "constant"
        value = 100
        input_data = np.random.rand(*input_shape).astype(self.dtype)
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            input_data = (
                np.random.rand(*input_shape) + 1j * np.random.rand(*input_shape)
            ).astype(self.dtype)
        np_out1 = self._get_numpy_out(
            input_data, pad, mode, value, data_format="NCDHW"
        )
        np_out2 = self._get_numpy_out(
            input_data, pad, mode, value, data_format="NDHWC"
        )
        np_out3 = self._get_numpy_out(
            input_data, pad_3, mode, value, data_format="NCDHW"
        )
        tensor_data = paddle.to_tensor(input_data)

        y1 = F.pad(
            tensor_data, pad=pad, mode=mode, value=value, data_format="NCDHW"
        )
        y2 = F.pad(
            tensor_data, pad=pad, mode=mode, value=value, data_format="NDHWC"
        )
        y3 = F.pad(
            tensor_data, pad=pad_3, mode=mode, value=value, data_format="NCDHW"
        )
        y4 = F.pad(tensor_data, pad=pad, mode=mode, value=value)

        np.testing.assert_allclose(y1.numpy(), np_out1, rtol=1e-05)
        np.testing.assert_allclose(y2.numpy(), np_out2, rtol=1e-05)
        np.testing.assert_allclose(y3.numpy(), np_out3, rtol=1e-05)
        np.testing.assert_allclose(y4.numpy(), np_out1, rtol=1e-05)

    def test_dygraph_2(self):
        paddle.disable_static()
        input_shape = (2, 3, 4, 5)
        pad = [1, 1, 3, 4]
        pad_3 = [1, 2, 1, 1, 3, 4, 5, 6]
        mode = "constant"
        value = 100
        input_data = np.random.rand(*input_shape).astype(self.dtype)
        np_out1 = self._get_numpy_out(
            input_data, pad, mode, value, data_format="NCHW"
        )
        np_out2 = self._get_numpy_out(
            input_data, pad, mode, value, data_format="NHWC"
        )
        np_out3 = self._get_numpy_out(
            input_data, pad_3, mode, value, data_format="NCHW"
        )

        tensor_data = paddle.to_tensor(input_data)
        tensor_pad = paddle.to_tensor(pad, dtype="int32")

        y1 = F.pad(
            tensor_data,
            pad=tensor_pad,
            mode=mode,
            value=value,
            data_format="NCHW",
        )
        y2 = F.pad(
            tensor_data,
            pad=tensor_pad,
            mode=mode,
            value=value,
            data_format="NHWC",
        )
        y3 = F.pad(
            tensor_data, pad=pad_3, mode=mode, value=value, data_format="NCHW"
        )
        y4 = F.pad(
            tensor_data,
            pad=tensor_pad,
            mode=mode,
            value=value,
        )
        np.testing.assert_allclose(y1.numpy(), np_out1, rtol=1e-05)
        np.testing.assert_allclose(y2.numpy(), np_out2, rtol=1e-05)
        np.testing.assert_allclose(y3.numpy(), np_out3, rtol=1e-05)
        np.testing.assert_allclose(y4.numpy(), np_out1, rtol=1e-05)

    def test_dygraph_3(self):
        paddle.disable_static()
        input_shape = (3, 4, 5)
        pad = [3, 4]
        pad_3 = [3, 4, 5, 6, 7, 8]
        mode = "constant"
        value = 100
        input_data = np.random.rand(*input_shape).astype(self.dtype)
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            input_data = (
                np.random.rand(*input_shape) + 1j * np.random.rand(*input_shape)
            ).astype(self.dtype)
        np_out1 = self._get_numpy_out(
            input_data, pad, mode, value, data_format="NCL"
        )
        np_out2 = self._get_numpy_out(
            input_data, pad, mode, value, data_format="NLC"
        )
        np_out3 = self._get_numpy_out(
            input_data, pad_3, mode, value, data_format="NCL"
        )
        tensor_data = paddle.to_tensor(input_data)
        tensor_pad = paddle.to_tensor(pad, dtype="int32")

        y1 = F.pad(
            tensor_data,
            pad=tensor_pad,
            mode=mode,
            value=value,
            data_format="NCL",
        )
        y2 = F.pad(
            tensor_data,
            pad=tensor_pad,
            mode=mode,
            value=value,
            data_format="NLC",
        )
        y3 = F.pad(
            tensor_data, pad=pad_3, mode=mode, value=value, data_format="NCL"
        )
        y4 = F.pad(
            tensor_data,
            pad=tensor_pad,
            mode=mode,
            value=value,
        )

        np.testing.assert_allclose(y1.numpy(), np_out1, rtol=1e-05)
        np.testing.assert_allclose(y2.numpy(), np_out2, rtol=1e-05)
        np.testing.assert_allclose(y3.numpy(), np_out3, rtol=1e-05)
        np.testing.assert_allclose(y4.numpy(), np_out1, rtol=1e-05)


class TestPadAPI_complex64(TestPadAPI):
    def init_dtype(self):
        self.dtype = np.complex64


class TestPadAPI_complex128(TestPadAPI):
    def init_dtype(self):
        self.dtype = np.complex128


class TestPad1dAPI(unittest.TestCase):
    def _get_numpy_out(
        self, input_data, pad, mode, value=0.0, data_format="NCL"
    ):
        if data_format == "NCL":
            pad = [
                (0, 0),
                (0, 0),
                (pad[0], pad[1]),
            ]
        else:
            pad = [
                (0, 0),
                (pad[0], pad[1]),
                (0, 0),
            ]

        if mode == "constant":
            out = np.pad(input_data, pad, mode=mode, constant_values=value)
        elif mode == "reflect":
            out = np.pad(input_data, pad, mode=mode)
        elif mode == "replicate":
            out = np.pad(input_data, pad, mode="edge")
        elif mode == "circular":
            out = np.pad(input_data, pad, mode="wrap")

        return out

    def setUp(self):
        self.init_dtype()
        self.places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.places.append(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def init_dtype(self):
        self.dtype = np.float32

    def test_class(self):
        paddle.disable_static()
        for place in self.places:
            input_shape = (3, 4, 5)
            pad = [1, 2]
            pad_int = 1
            value = 100
            input_data = np.random.rand(*input_shape).astype(self.dtype)
            if self.dtype == np.complex64 or self.dtype == np.complex128:
                input_data = (
                    np.random.rand(*input_shape)
                    + 1j * np.random.rand(*input_shape)
                ).astype(self.dtype)
            pad_reflection = nn.Pad1D(padding=pad, mode="reflect")
            pad_replication = nn.Pad1D(padding=pad, mode="replicate")
            pad_constant = nn.Pad1D(padding=pad, mode="constant", value=value)
            pad_constant_int = nn.Pad1D(
                padding=pad_int, mode="constant", value=value
            )
            pad_circular = nn.Pad1D(padding=pad, mode="circular")

            data = paddle.to_tensor(input_data)

            output = pad_reflection(data)
            np_out = self._get_numpy_out(
                input_data, pad, "reflect", data_format="NCL"
            )
            np.testing.assert_allclose(output.numpy(), np_out, rtol=1e-05)

            output = pad_replication(data)
            np_out = self._get_numpy_out(
                input_data, pad, "replicate", data_format="NCL"
            )
            np.testing.assert_allclose(output.numpy(), np_out, rtol=1e-05)

            output = pad_constant(data)
            np_out = self._get_numpy_out(
                input_data, pad, "constant", value=value, data_format="NCL"
            )
            np.testing.assert_allclose(output.numpy(), np_out, rtol=1e-05)

            output = pad_constant_int(data)
            np_out = self._get_numpy_out(
                input_data,
                [pad_int] * 2,
                "constant",
                value=value,
                data_format="NCL",
            )
            np.testing.assert_allclose(output.numpy(), np_out, rtol=1e-05)

            output = pad_circular(data)
            np_out = self._get_numpy_out(
                input_data, pad, "circular", value=value, data_format="NCL"
            )
            np.testing.assert_allclose(output.numpy(), np_out, rtol=1e-05)


class TestPad1dAPI_complex64(TestPad1dAPI):
    def init_dtype(self):
        self.dtype = np.complex64


class TestPad1dAPI_complex128(TestPad1dAPI):
    def init_dtype(self):
        self.dtype = np.complex128


class TestPad2dAPI(unittest.TestCase):
    def _get_numpy_out(
        self, input_data, pad, mode, value=0.0, data_format="NCHW"
    ):
        if data_format == "NCHW":
            pad = [
                (0, 0),
                (0, 0),
                (pad[2], pad[3]),
                (pad[0], pad[1]),
            ]
        else:
            pad = [
                (0, 0),
                (pad[2], pad[3]),
                (pad[0], pad[1]),
                (0, 0),
            ]

        if mode == "constant":
            out = np.pad(input_data, pad, mode=mode, constant_values=value)
        elif mode == "reflect":
            out = np.pad(input_data, pad, mode=mode)
        elif mode == "replicate":
            out = np.pad(input_data, pad, mode="edge")
        elif mode == "circular":
            out = np.pad(input_data, pad, mode="wrap")

        return out

    def setUp(self):
        self.init_dtype()
        self.places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.places.append(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def init_dtype(self):
        self.dtype = np.float32

    def test_class(self):
        paddle.disable_static()
        for place in self.places:
            input_shape = (3, 4, 5, 6)
            pad = [1, 2, 2, 1]
            pad_int = 1
            value = 100
            input_data = np.random.rand(*input_shape).astype(self.dtype)
            if self.dtype == np.complex64 or self.dtype == np.complex128:
                input_data = (
                    np.random.rand(*input_shape)
                    + 1j * np.random.rand(*input_shape)
                ).astype(self.dtype)
            pad_reflection = nn.Pad2D(padding=pad, mode="reflect")
            pad_replication = nn.Pad2D(padding=pad, mode="replicate")
            pad_constant = nn.Pad2D(padding=pad, mode="constant", value=value)
            pad_constant_int = nn.Pad2D(
                padding=pad_int, mode="constant", value=value
            )
            pad_circular = nn.Pad2D(padding=pad, mode="circular")

            data = paddle.to_tensor(input_data)

            output = pad_reflection(data)
            np_out = self._get_numpy_out(
                input_data, pad, "reflect", data_format="NCHW"
            )
            np.testing.assert_allclose(output.numpy(), np_out, rtol=1e-05)

            output = pad_replication(data)
            np_out = self._get_numpy_out(
                input_data, pad, "replicate", data_format="NCHW"
            )
            np.testing.assert_allclose(output.numpy(), np_out, rtol=1e-05)

            output = pad_constant(data)
            np_out = self._get_numpy_out(
                input_data, pad, "constant", value=value, data_format="NCHW"
            )
            np.testing.assert_allclose(output.numpy(), np_out, rtol=1e-05)

            output = pad_constant_int(data)
            np_out = self._get_numpy_out(
                input_data,
                [pad_int] * 4,
                "constant",
                value=value,
                data_format="NCHW",
            )
            np.testing.assert_allclose(output.numpy(), np_out, rtol=1e-05)

            output = pad_circular(data)
            np_out = self._get_numpy_out(
                input_data, pad, "circular", data_format="NCHW"
            )
            np.testing.assert_allclose(output.numpy(), np_out, rtol=1e-05)


class TestPad2dAPI_complex64(TestPad2dAPI):
    def init_dtype(self):
        self.dtype = np.complex64


class TestPad2dAPI_complex128(TestPad2dAPI):
    def init_dtype(self):
        self.dtype = np.complex128


class TestPad3dAPI(unittest.TestCase):
    def _get_numpy_out(
        self, input_data, pad, mode, value=0.0, data_format="NCDHW"
    ):
        if data_format == "NCDHW":
            pad = [
                (0, 0),
                (0, 0),
                (pad[4], pad[5]),
                (pad[2], pad[3]),
                (pad[0], pad[1]),
            ]
        else:
            pad = [
                (0, 0),
                (pad[4], pad[5]),
                (pad[2], pad[3]),
                (pad[0], pad[1]),
                (0, 0),
            ]

        if mode == "constant":
            out = np.pad(input_data, pad, mode=mode, constant_values=value)
        elif mode == "reflect":
            out = np.pad(input_data, pad, mode=mode)
        elif mode == "replicate":
            out = np.pad(input_data, pad, mode="edge")
        elif mode == "circular":
            out = np.pad(input_data, pad, mode="wrap")

        return out

    def setUp(self):
        self.init_dtype()
        self.places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.places.append(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def init_dtype(self):
        self.dtype = np.float32

    def test_class(self):
        paddle.disable_static()
        for place in self.places:
            input_shape = (3, 4, 5, 6, 7)
            pad = [1, 2, 2, 1, 1, 0]
            pad_int = 1
            value = 100
            input_data = np.random.rand(*input_shape).astype(self.dtype)
            if self.dtype == np.complex64 or self.dtype == np.complex128:
                input_data = (
                    np.random.rand(*input_shape)
                    + 1j * np.random.rand(*input_shape)
                ).astype(self.dtype)
            pad_reflection = nn.Pad3D(padding=pad, mode="reflect")
            pad_replication = nn.Pad3D(padding=pad, mode="replicate")
            pad_constant = nn.Pad3D(padding=pad, mode="constant", value=value)
            pad_constant_int = nn.Pad3D(
                padding=pad_int, mode="constant", value=value
            )
            pad_circular = nn.Pad3D(padding=pad, mode="circular")

            data = paddle.to_tensor(input_data)

            output = pad_reflection(data)
            np_out = self._get_numpy_out(
                input_data, pad, "reflect", data_format="NCDHW"
            )
            np.testing.assert_allclose(output.numpy(), np_out, rtol=1e-05)

            output = pad_replication(data)
            np_out = self._get_numpy_out(
                input_data, pad, "replicate", data_format="NCDHW"
            )
            np.testing.assert_allclose(output.numpy(), np_out, rtol=1e-05)

            output = pad_constant(data)
            np_out = self._get_numpy_out(
                input_data, pad, "constant", value=value, data_format="NCDHW"
            )
            np.testing.assert_allclose(output.numpy(), np_out, rtol=1e-05)

            output = pad_constant_int(data)
            np_out = self._get_numpy_out(
                input_data,
                [pad_int] * 6,
                "constant",
                value=value,
                data_format="NCDHW",
            )
            np.testing.assert_allclose(output.numpy(), np_out, rtol=1e-05)

            output = pad_circular(data)
            np_out = self._get_numpy_out(
                input_data, pad, "circular", data_format="NCDHW"
            )
            np.testing.assert_allclose(output.numpy(), np_out, rtol=1e-05)

    def test_pad_tensor(self):
        paddle.disable_static()
        for place in self.places:
            input_shape = (3, 4, 5, 6, 7)
            pad = [1, 2, 2, 1, 1, 0]
            pad_tensor = paddle.to_tensor(pad)
            input_data = np.random.rand(*input_shape).astype(self.dtype)
            if self.dtype == np.complex64 or self.dtype == np.complex128:
                input_data = (
                    np.random.rand(*input_shape)
                    + 1j * np.random.rand(*input_shape)
                ).astype(self.dtype)
            pad_reflection_ncdhw = nn.Pad3D(
                padding=pad_tensor, mode="reflect", data_format="NCDHW"
            )
            pad_reflection_ndhwc = nn.Pad3D(
                padding=pad_tensor, mode="reflect", data_format="NDHWC"
            )
            data = paddle.to_tensor(input_data)

            output = pad_reflection_ncdhw(data)
            np_out = self._get_numpy_out(
                input_data, pad, "reflect", data_format="NCDHW"
            )
            np.testing.assert_allclose(output.numpy(), np_out, rtol=1e-05)

            output = pad_reflection_ndhwc(data)
            np_out = self._get_numpy_out(
                input_data, pad, "reflect", data_format="NDHWC"
            )
            np.testing.assert_allclose(output.numpy(), np_out, rtol=1e-05)


class TestPad3dAPI_complex64(TestPad3dAPI):
    def init_dtype(self):
        self.dtype = np.complex64


class TestPad3dAPI_complex128(TestPad3dAPI):
    def init_dtype(self):
        self.dtype = np.complex128


class TestPad3dOpError(unittest.TestCase):
    def setUp(self):
        self.init_dtype()
        self.places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.places.append(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def init_dtype(self):
        self.dtype = np.float32

    def test_errors(self):
        def test_variable():
            input_shape = (1, 2, 3, 4, 5)
            data = np.random.rand(*input_shape).astype(self.dtype)
            if self.dtype == np.complex64 or self.dtype == np.complex128:
                data = (
                    np.random.rand(*input_shape)
                    + 1j * np.random.rand(*input_shape)
                ).astype(self.dtype)
            y = F.pad(x=data, pad=[1, 1, 1, 1, 1, 1], data_format="NCDHW")

        def test_reflect_1():
            input_shape = (1, 2, 3, 4, 5)
            data = np.random.rand(*input_shape).astype(self.dtype)
            if self.dtype == np.complex64 or self.dtype == np.complex128:
                data = (
                    np.random.rand(*input_shape)
                    + 1j * np.random.rand(*input_shape)
                ).astype(self.dtype)
            x = paddle.to_tensor(data)
            y = F.pad(
                x,
                pad=[5, 6, 1, 1, 1, 1],
                value=1,
                mode='reflect',
                data_format="NCDHW",
            )

        def test_reflect_2():
            input_shape = (1, 2, 3, 4, 5)
            data = np.random.rand(*input_shape).astype(self.dtype)
            if self.dtype == np.complex64 or self.dtype == np.complex128:
                data = (
                    np.random.rand(*input_shape)
                    + 1j * np.random.rand(*input_shape)
                ).astype(self.dtype)
            x = paddle.to_tensor(data)
            y = F.pad(
                x,
                pad=[1, 1, 4, 3, 1, 1],
                value=1,
                mode='reflect',
                data_format="NCDHW",
            )

        def test_reflect_3():
            input_shape = (1, 2, 3, 4, 5)
            data = np.random.rand(*input_shape).astype(self.dtype)
            if self.dtype == np.complex64 or self.dtype == np.complex128:
                data = (
                    np.random.rand(*input_shape)
                    + 1j * np.random.rand(*input_shape)
                ).astype(self.dtype)
            x = paddle.to_tensor(data)
            y = F.pad(
                x,
                pad=[1, 1, 1, 1, 2, 3],
                value=1,
                mode='reflect',
                data_format="NCDHW",
            )

        def test_circular_1():
            input_shape = (1, 2, 0, 4, 5)
            data = np.random.rand(*input_shape).astype(self.dtype)
            if self.dtype == np.complex64 or self.dtype == np.complex128:
                data = (
                    np.random.rand(*input_shape)
                    + 1j * np.random.rand(*input_shape)
                ).astype(self.dtype)
            x = paddle.to_tensor(data)
            y = F.pad(
                x, pad=[1, 1, 1, 1, 2, 3], mode='circular', data_format="NCDHW"
            )

        def test_replicate_1():
            input_shape = (1, 2, 0, 4, 5)
            data = np.random.rand(*input_shape).astype(self.dtype)
            if self.dtype == np.complex64 or self.dtype == np.complex128:
                data = (
                    np.random.rand(*input_shape)
                    + 1j * np.random.rand(*input_shape)
                ).astype(self.dtype)
            x = paddle.to_tensor(data)
            y = F.pad(
                x, pad=[1, 1, 1, 1, 2, 3], mode='replicate', data_format="NCDHW"
            )

        paddle.disable_static()
        for place in self.places:
            self.assertRaises(ValueError, test_variable)
            self.assertRaises(Exception, test_reflect_1)
            self.assertRaises(Exception, test_reflect_2)
            self.assertRaises(Exception, test_reflect_3)
            self.assertRaises(Exception, test_circular_1)
            self.assertRaises(Exception, test_replicate_1)
        paddle.enable_static()


class TestPad3dOpError_complex64(TestPad3dOpError):
    def init_dtype(self):
        self.dtype = np.complex64


class TestPad3dOpError_complex128(TestPad3dOpError):
    def init_dtype(self):
        self.dtype = np.complex128


class TestPadDataformatError(unittest.TestCase):
    def test_errors(self):
        def test_ncl():
            input_shape = (1, 2, 3, 4)
            pad = paddle.to_tensor(np.array([2, 1, 2, 1]).astype('int32'))
            data = (
                np.arange(np.prod(input_shape), dtype=np.float64).reshape(
                    input_shape
                )
                + 1
            )
            my_pad = nn.Pad1D(padding=pad, mode="replicate", data_format="NCL")
            data = paddle.to_tensor(data)
            result = my_pad(data)

        def test_nchw():
            input_shape = (1, 2, 4)
            pad = paddle.to_tensor(np.array([2, 1, 2, 1]).astype('int32'))
            data = (
                np.arange(np.prod(input_shape), dtype=np.float64).reshape(
                    input_shape
                )
                + 1
            )
            my_pad = nn.Pad1D(padding=pad, mode="replicate", data_format="NCHW")
            data = paddle.to_tensor(data)
            result = my_pad(data)

        def test_ncdhw():
            input_shape = (1, 2, 3, 4)
            pad = paddle.to_tensor(np.array([2, 1, 2, 1]).astype('int32'))
            data = (
                np.arange(np.prod(input_shape), dtype=np.float64).reshape(
                    input_shape
                )
                + 1
            )
            my_pad = nn.Pad1D(
                padding=pad, mode="replicate", data_format="NCDHW"
            )
            data = paddle.to_tensor(data)
            result = my_pad(data)

        self.assertRaises(AssertionError, test_ncl)

        self.assertRaises(AssertionError, test_nchw)

        self.assertRaises(AssertionError, test_ncdhw)


if __name__ == '__main__':
    unittest.main()
