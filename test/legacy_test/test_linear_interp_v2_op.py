#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import platform
import unittest

import numpy as np
from op_test import OpTest, convert_float_to_uint16, paddle_static_guard

import paddle
from paddle import base
from paddle.base import core
from paddle.nn.functional import interpolate


def create_test_case0(self):
    self.interp_method = 'linear'
    self.input_shape = [1, 3, 100]
    self.out_w = 50
    self.scale = 0.5
    self.out_size = np.array(
        [
            50,
        ]
    ).astype("int32")
    self.align_corners = False
    self.align_mode = 1


def linear_interp_test(
    x,
    OutSize=None,
    SizeTensor=None,
    Scale=None,
    data_layout='NCHW',
    out_d=-1,
    out_h=-1,
    out_w=-1,
    scale=[],
    interp_method='linear',
    align_corners=True,
    align_mode=0,
):
    if isinstance(scale, (float, int)):
        scale_list = []
        for _ in range(len(x.shape) - 2):
            scale_list.append(scale)
        scale = list(map(float, scale_list))
    elif isinstance(scale, (list, tuple)):
        scale = list(map(float, scale))
    if SizeTensor is not None:
        if not isinstance(SizeTensor, list) and not isinstance(
            SizeTensor, tuple
        ):
            SizeTensor = [SizeTensor]
    return paddle._C_ops.linear_interp(
        x,
        OutSize,
        SizeTensor,
        Scale,
        data_layout,
        out_d,
        out_h,
        out_w,
        scale,
        interp_method,
        align_corners,
        align_mode,
    )


def linear_interp_np(
    input,
    out_w,
    scale_w=0,
    out_size=None,
    actual_shape=None,
    align_corners=True,
    align_mode=0,
    data_layout='NCHW',
):
    if data_layout == "NHWC":
        input = np.transpose(input, (0, 2, 1))  # NHWC => NCHW
    if out_size is not None:
        out_w = out_size[0]
    if actual_shape is not None:
        out_w = actual_shape[0]
    batch_size, channel, in_w = input.shape

    ratio_w = 0.0
    if out_w > 1:
        if align_corners:
            ratio_w = (in_w - 1.0) / (out_w - 1.0)
        else:
            if scale_w > 0:
                ratio_w = 1.0 / scale_w
            else:
                ratio_w = 1.0 * in_w / out_w

    out = np.zeros((batch_size, channel, out_w))

    for j in range(out_w):
        if align_mode == 0 and not align_corners:
            w = int(ratio_w * (j + 0.5) - 0.5)
        else:
            w = int(ratio_w * j)
        w = max(0, w)
        wid = 1 if w < in_w - 1 else 0

        if align_mode == 0 and not align_corners:
            idx_src_w = max(ratio_w * (j + 0.5) - 0.5, 0)
            w1lambda = idx_src_w - w
        else:
            w1lambda = ratio_w * j - w
        w2lambda = 1.0 - w1lambda

        out[:, :, j] = (
            w2lambda * input[:, :, w] + w1lambda * input[:, :, w + wid]
        )

    if data_layout == "NHWC":
        out = np.transpose(out, (0, 2, 1))  # NCHW => NHWC

    return out.astype(input.dtype)


class TestLinearInterpOp(OpTest):
    def setUp(self):
        self.python_api = linear_interp_test
        self.out_size = None
        self.actual_shape = None
        self.data_layout = 'NCHW'
        self.dtype = np.float64
        self.init_test_case()
        self.op_type = "linear_interp_v2"
        input_np = np.random.random(self.input_shape).astype(self.dtype)

        scale_w = 0
        if self.data_layout == "NCHW":
            in_w = self.input_shape[2]
        else:
            in_w = self.input_shape[1]

        if self.scale > 0:
            if isinstance(self.scale, (float, int)):
                self.scale = float(self.scale)
            if isinstance(self.scale, list):
                self.scale = float(self.scale[0])
            out_w = int(in_w * self.scale)
        else:
            out_w = self.out_w

        output_np = linear_interp_np(
            input_np,
            out_w,
            self.scale,
            self.out_size,
            self.actual_shape,
            self.align_corners,
            self.align_mode,
            self.data_layout,
        )
        self.inputs = {'X': input_np}
        if self.out_size is not None:
            self.inputs['OutSize'] = self.out_size
        if self.actual_shape is not None:
            self.inputs['OutSize'] = self.actual_shape

        self.attrs = {
            'out_w': self.out_w,
            'interp_method': self.interp_method,
            'align_corners': self.align_corners,
            'align_mode': self.align_mode,
            'data_layout': self.data_layout,
        }
        if self.scale > 0:
            if isinstance(self.scale, (float, int)):
                self.scale = [float(self.scale)]
            self.attrs['scale'] = self.scale
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        if platform.system() == "Linux":
            self.check_output(atol=1e-7, check_pir=True)
        else:
            self.check_output(atol=1e-5, check_pir=True)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', in_place=True, check_pir=True)

    def init_test_case(self):
        create_test_case0(self)


class TestLinearInterpOpDataLayout(TestLinearInterpOp):
    def init_test_case(self):
        self.interp_method = 'linear'
        self.input_shape = [1, 100, 3]
        self.out_w = 50
        self.scale = 0.5
        self.out_size = np.array(
            [
                50,
            ]
        ).astype("int32")
        self.align_corners = False
        self.align_mode = 1
        self.data_layout = 'NHWC'


class TestLinearInterpOpAlignMode(TestLinearInterpOp):
    def init_test_case(self):
        self.interp_method = 'linear'
        self.input_shape = [1, 3, 100]
        self.out_w = 50
        self.scale = 0.5
        self.out_size = np.array(
            [
                50,
            ]
        ).astype("int32")
        self.align_corners = False
        self.align_mode = 0


class TestLinearInterpOpScale(TestLinearInterpOp):
    def init_test_case(self):
        self.interp_method = 'linear'
        self.input_shape = [1, 3, 100]
        self.out_w = 50
        self.scale = 0.8
        self.out_size = np.array(
            [
                50,
            ]
        ).astype("int32")
        self.align_corners = False
        self.align_mode = 0


class TestLinearInterpOpSizeTensor(TestLinearInterpOp):
    def setUp(self):
        self.python_api = linear_interp_test
        self.out_size = None
        self.actual_shape = None
        self.data_layout = 'NCHW'
        self.init_test_case()
        self.op_type = "linear_interp_v2"
        input_np = np.random.random(self.input_shape).astype("float64")
        self.shape_by_1Dtensor = False
        self.scale_by_1Dtensor = False

        if self.data_layout == "NCHW":
            in_w = self.input_shape[2]
        else:
            in_w = self.input_shape[1]

        if self.scale > 0:
            if isinstance(self.scale, (float, int)):
                self.scale = float(self.scale)
            if isinstance(self.scale, list):
                self.scale = float(self.scale[0])
            out_w = int(in_w * self.scale)
        else:
            out_w = self.out_w

        output_np = linear_interp_np(
            input_np,
            out_w,
            0,
            self.out_size,
            self.actual_shape,
            self.align_corners,
            self.align_mode,
            self.data_layout,
        )

        self.inputs = {'X': input_np}
        if self.out_size is not None and self.shape_by_1Dtensor:
            self.inputs['OutSize'] = self.out_size
        elif self.actual_shape is not None and self.shape_by_1Dtensor:
            self.inputs['OutSize'] = self.actual_shape
        else:
            size_tensor = []
            for index, ele in enumerate(self.out_size):
                size_tensor.append(
                    ("x" + str(index), np.ones(1).astype('int32') * ele)
                )
            self.inputs['SizeTensor'] = size_tensor

        self.attrs = {
            'out_w': self.out_w,
            'interp_method': self.interp_method,
            'align_corners': self.align_corners,
            'align_mode': self.align_mode,
            'data_layout': self.data_layout,
        }
        if self.scale > 0:
            if isinstance(self.scale, (float, int)):
                self.scale = [self.scale]
            if isinstance(self.scale, list) and len(self.scale) == 1:
                self.scale = [self.scale[0], self.scale[0]]
            self.attrs['scale'] = self.scale
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        if platform.system() == "Linux":
            self.check_output(atol=1e-7, check_pir=False)
        else:
            self.check_output(atol=1e-5, check_pir=False)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', in_place=True, check_pir=True)


class TestLinearInterpOpAPI2_0(unittest.TestCase):
    def test_case(self):
        # dygraph
        x_data = np.random.random((1, 3, 128)).astype("float32")
        us_1 = paddle.nn.Upsample(
            size=[64],
            mode='linear',
            align_mode=1,
            align_corners=False,
        )
        with base.dygraph.guard():
            x = paddle.to_tensor(x_data)
            interp = us_1(x)

            expect = linear_interp_np(
                x_data, out_w=64, align_mode=1, align_corners=False
            )

            np.testing.assert_allclose(interp.numpy(), expect, rtol=1e-05)


class TestLinearInterpOpAPI2_0_case2(unittest.TestCase):
    def test_case(self):
        # dygraph
        x_data = np.random.random((1, 3, 128)).astype("float32")
        with base.dygraph.guard():
            x = paddle.to_tensor(x_data)
            interp = interpolate(
                x,
                size=[64],
                mode='linear',
                align_mode=1,
                align_corners=False,
            )
            expect = linear_interp_np(
                x_data, out_w=64, align_mode=1, align_corners=False
            )

            np.testing.assert_allclose(interp.numpy(), expect, rtol=1e-05)


class TestLinearInterpOpFP16(TestLinearInterpOp):
    def test_check_output(self):
        self.check_output(atol=1e-3, check_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['X'], 'Out', in_place=True, max_relative_error=1e-2, check_pir=True
        )

    def init_test_case(self):
        create_test_case0(self)
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support the bfloat16",
)
class TestLinearInterpOpBF16(OpTest):
    def setUp(self):
        self.python_api = linear_interp_test
        self.out_size = None
        self.actual_shape = None
        self.data_layout = 'NCHW'
        self.init_test_case()
        self.op_type = "linear_interp_v2"
        self.dtype = np.uint16
        input_np = np.random.random(self.input_shape).astype("float32")

        scale_w = 0
        if self.data_layout == "NCHW":
            in_w = self.input_shape[2]
        else:
            in_w = self.input_shape[1]

        if self.scale > 0:
            if isinstance(self.scale, (float, int)):
                self.scale = float(self.scale)
            if isinstance(self.scale, list):
                self.scale = float(self.scale[0])
            out_w = int(in_w * self.scale)
        else:
            out_w = self.out_w

        output_np = linear_interp_np(
            input_np,
            out_w,
            self.scale,
            self.out_size,
            self.actual_shape,
            self.align_corners,
            self.align_mode,
            self.data_layout,
        )
        self.inputs = {'X': convert_float_to_uint16(input_np)}
        if self.out_size is not None:
            self.inputs['OutSize'] = self.out_size
        if self.actual_shape is not None:
            self.inputs['OutSize'] = self.actual_shape

        self.attrs = {
            'out_w': self.out_w,
            'interp_method': self.interp_method,
            'align_corners': self.align_corners,
            'align_mode': self.align_mode,
            'data_layout': self.data_layout,
        }
        if self.scale > 0:
            if isinstance(self.scale, (float, int)):
                self.scale = [float(self.scale)]
            self.attrs['scale'] = self.scale
        self.outputs = {'Out': convert_float_to_uint16(output_np)}

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, atol=1e-2, check_pir=True)

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place,
            ['X'],
            'Out',
            in_place=True,
            max_relative_error=1e-2,
            check_pir=True,
        )

    def init_test_case(self):
        create_test_case0(self)


class TestResizeLinearOpUint8(OpTest):
    def setUp(self):
        self.out_size = None
        self.actual_shape = None
        self.init_test_case()
        self.op_type = "linear_interp_v2"
        self.python_api = linear_interp_test
        input_np = np.random.random(self.input_shape).astype("uint8")

        if self.scale > 0:
            if isinstance(self.scale, (float, int)):
                self.scale = float(self.scale)
            if isinstance(self.scale, list):
                self.scale = float(self.scale[0])
            out_w = int(self.input_shape[2] * self.scale)
        else:
            out_w = self.out_w

        output_np = linear_interp_np(
            input_np,
            out_w,
            0,
            self.out_size,
            self.actual_shape,
            self.align_corners,
            self.align_mode,
        )
        self.inputs = {'X': input_np}
        if self.out_size is not None:
            self.inputs['OutSize'] = self.out_size

        self.attrs = {
            'out_w': self.out_w,
            'interp_method': self.interp_method,
            'align_corners': self.align_corners,
            'align_mode': self.align_mode,
        }
        if self.scale > 0:
            if isinstance(self.scale, (float, int)):
                self.scale = [self.scale]
            if isinstance(self.scale, list) and len(self.scale) == 1:
                self.scale = [self.scale[0], self.scale[0]]
            self.attrs['scale'] = self.scale
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        if platform.system() == "Linux":
            self.check_output_with_place(
                place=core.CPUPlace(), atol=1e-7, check_pir=True
            )
        else:
            self.check_output_with_place(
                place=core.CPUPlace(), atol=1e-5, check_pir=True
            )

    def init_test_case(self):
        self.interp_method = 'linear'
        self.input_shape = [2, 3, 100]
        self.out_w = 50
        self.scale = 0.0
        self.out_size = np.array(
            [
                50,
            ]
        ).astype("int32")
        self.align_corners = True
        self.align_mode = 1


class TestLinearInterpOpError(unittest.TestCase):

    def test_error(self):
        with paddle_static_guard():
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):

                def input_shape_error():
                    x1 = paddle.static.data(
                        name="x1", shape=[1], dtype="float32"
                    )
                    out1 = paddle.nn.Upsample(
                        size=[256], data_format='NCW', mode='linear'
                    )
                    out1_res = out1(x1)

                def data_format_error():
                    x2 = paddle.static.data(
                        name="x2", shape=[1, 3, 128], dtype="float32"
                    )
                    out2 = paddle.nn.Upsample(
                        size=[256], data_format='NHWCD', mode='linear'
                    )
                    out2_res = out2(x2)

                def out_shape_error():
                    x3 = paddle.static.data(
                        name="x3", shape=[1, 3, 128], dtype="float32"
                    )
                    out3 = paddle.nn.Upsample(
                        size=[256, 256], data_format='NHWC', mode='linear'
                    )
                    out3_res = out3(x3)

                self.assertRaises(ValueError, input_shape_error)
                self.assertRaises(ValueError, data_format_error)
                self.assertRaises(ValueError, out_shape_error)


@unittest.skipIf(
    not base.core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestLinearInterpOpForFloat16(unittest.TestCase):
    def init_test_case(self):
        self.interp_method = 'linear'
        self.input_shape = [1, 3, 64]
        self.scale = 2
        self.align_corners = False
        self.align_mode = 1
        self.data_layout = 'NCW'

    def check_main(self, x_np, dtype):
        paddle.disable_static()
        x_np = x_np.astype(dtype)
        x = paddle.to_tensor(x_np)
        x.stop_gradient = False
        y = interpolate(
            x,
            scale_factor=self.scale,
            mode=self.interp_method,
            align_mode=self.align_mode,
            align_corners=self.align_corners,
            data_format=self.data_layout,
        )
        x_g = paddle.grad(y, x)
        y_np = y[0].numpy().astype('float32')
        x_g_np = x_g[0].numpy().astype('float32')
        paddle.enable_static()
        return y_np, x_g_np

    def test_main(self):
        self.init_test_case()
        x_np = np.random.random(self.input_shape).astype("float16")

        y_np_1, x_g_np_1 = self.check_main(x_np, 'float16')
        y_np_2, x_g_np_2 = self.check_main(x_np, 'float32')
        # forward
        np.testing.assert_allclose(y_np_1, y_np_2, rtol=1e-03)
        # backward
        np.testing.assert_allclose(x_g_np_1, x_g_np_2)


if __name__ == "__main__":
    unittest.main()
