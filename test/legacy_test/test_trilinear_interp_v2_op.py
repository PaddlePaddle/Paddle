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
from paddle import base
from paddle.base import core
from paddle.nn.functional import interpolate

np.random.seed(123)


def create_test_case0(self):
    self.interp_method = 'trilinear'
    self.input_shape = [2, 3, 4, 4, 4]
    self.out_d = 2
    self.out_h = 2
    self.out_w = 2
    self.scale = []
    self.out_size = np.array([3, 3, 3]).astype("int32")
    self.align_corners = True
    self.align_mode = 1


def create_test_case1(self):
    self.interp_method = 'trilinear'
    self.input_shape = [2, 1, 7, 8, 9]
    self.out_d = 1
    self.out_h = 1
    self.out_w = 1
    self.scale = []
    self.align_corners = True
    self.align_mode = 1


def create_test_case2(self):
    self.interp_method = 'trilinear'
    self.input_shape = [2, 3, 9, 6, 8]
    self.out_d = 12
    self.out_h = 12
    self.out_w = 12
    self.scale = []
    self.align_corners = True
    self.align_mode = 1


def create_test_case3(self):
    self.interp_method = 'trilinear'
    self.input_shape = [3, 2, 16, 8, 4]
    self.out_d = 32
    self.out_h = 16
    self.out_w = 8
    self.scale = []
    self.align_corners = True
    self.align_mode = 1


def create_test_case4(self):
    self.interp_method = 'trilinear'
    self.input_shape = [4, 1, 7, 8, 9]
    self.out_d = 1
    self.out_h = 1
    self.out_w = 1
    self.scale = []
    self.out_size = np.array([2, 2, 2]).astype("int32")
    self.align_corners = True
    self.align_mode = 1


def create_test_case5(self):
    self.interp_method = 'trilinear'
    self.input_shape = [3, 3, 9, 6, 8]
    self.out_d = 12
    self.out_h = 12
    self.out_w = 12
    self.scale = []
    self.out_size = np.array([11, 11, 11]).astype("int32")
    self.align_corners = True
    self.align_mode = 1


def create_test_case6(self):
    self.interp_method = 'trilinear'
    self.input_shape = [1, 1, 16, 8, 4]
    self.out_d = 8
    self.out_h = 32
    self.out_w = 16
    self.scale = []
    self.out_size = np.array([17, 9, 5]).astype("int32")
    self.align_corners = True
    self.align_mode = 1


def trilinear_interp_test(
    x,
    OutSize=None,
    SizeTensor=None,
    Scale=None,
    data_layout='NCHW',
    out_d=-1,
    out_h=-1,
    out_w=-1,
    scale=[],
    interp_method='trilinear',
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
    return paddle._C_ops.trilinear_interp(
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


def trilinear_interp_np(
    input,
    out_d,
    out_h,
    out_w,
    scale_d=0,
    scale_h=0,
    scale_w=0,
    out_size=None,
    actual_shape=None,
    align_corners=True,
    align_mode=0,
    data_layout='NCDHW',
):
    """trilinear interpolation implement in shape [N, C, D, H, W]"""
    if data_layout == "NDHWC":
        input = np.transpose(input, (0, 4, 1, 2, 3))  # NDHWC => NCDHW
    if out_size is not None:
        out_d = out_size[0]
        out_h = out_size[1]
        out_w = out_size[2]
    if actual_shape is not None:
        out_d = actual_shape[0]
        out_h = actual_shape[1]
        out_w = actual_shape[2]
    batch_size, channel, in_d, in_h, in_w = input.shape

    ratio_d = ratio_h = ratio_w = 0.0
    if out_d > 1:
        if align_corners:
            ratio_d = (in_d - 1.0) / (out_d - 1.0)
        else:
            if scale_d > 0:
                ratio_d = 1.0 / scale_d
            else:
                ratio_d = 1.0 * in_d / out_d
    if out_h > 1:
        if align_corners:
            ratio_h = (in_h - 1.0) / (out_h - 1.0)
        else:
            if scale_h > 0:
                ratio_h = 1.0 / scale_h
            else:
                ratio_h = 1.0 * in_h / out_h
    if out_w > 1:
        if align_corners:
            ratio_w = (in_w - 1.0) / (out_w - 1.0)
        else:
            if scale_w > 0:
                ratio_w = 1.0 / scale_w
            else:
                ratio_w = 1.0 * in_w / out_w

    out = np.zeros((batch_size, channel, out_d, out_h, out_w))

    for i in range(out_d):
        if align_mode == 0 and not align_corners:
            d = int(ratio_d * (i + 0.5) - 0.5)
        else:
            d = int(ratio_d * i)

        d = max(0, d)
        did = 1 if d < in_d - 1 else 0
        if align_mode == 0 and not align_corners:
            idx_src_d = max(ratio_d * (i + 0.5) - 0.5, 0)
            d1lambda = idx_src_d - d
        else:
            d1lambda = ratio_d * i - d
        d2lambda = 1.0 - d1lambda

        for j in range(out_h):
            if align_mode == 0 and not align_corners:
                h = int(ratio_h * (j + 0.5) - 0.5)
            else:
                h = int(ratio_h * j)

            h = max(0, h)
            hid = 1 if h < in_h - 1 else 0
            if align_mode == 0 and not align_corners:
                idx_src_h = max(ratio_h * (j + 0.5) - 0.5, 0)
                h1lambda = idx_src_h - h
            else:
                h1lambda = ratio_h * j - h
            h2lambda = 1.0 - h1lambda

            for k in range(out_w):
                if align_mode == 0 and not align_corners:
                    w = int(ratio_w * (k + 0.5) - 0.5)
                else:
                    w = int(ratio_w * k)
                w = max(0, w)
                wid = 1 if w < in_w - 1 else 0
                if align_mode == 0 and not align_corners:
                    idx_src_w = max(ratio_w * (k + 0.5) - 0.5, 0)
                    w1lambda = idx_src_w - w
                else:
                    w1lambda = ratio_w * k - w
                w2lambda = 1.0 - w1lambda

                out[:, :, i, j, k] = d2lambda * (
                    h2lambda
                    * (
                        w2lambda * input[:, :, d, h, w]
                        + w1lambda * input[:, :, d, h, w + wid]
                    )
                    + h1lambda
                    * (
                        w2lambda * input[:, :, d, h + hid, w]
                        + w1lambda * input[:, :, d, h + hid, w + wid]
                    )
                ) + d1lambda * (
                    h2lambda
                    * (
                        w2lambda * input[:, :, d + did, h, w]
                        + w1lambda * input[:, :, d + did, h, w + wid]
                    )
                    + h1lambda
                    * (
                        w2lambda * input[:, :, d + did, h + hid, w]
                        + w1lambda * input[:, :, d + did, h + hid, w + wid]
                    )
                )
    if data_layout == "NDHWC":
        out = np.transpose(out, (0, 2, 3, 4, 1))  # NCDHW => NDHWC

    return out.astype(input.dtype)


class TestTrilinearInterpOp(OpTest):
    def setUp(self):
        self.python_api = trilinear_interp_test
        self.out_size = None
        self.actual_shape = None
        self.data_layout = 'NCDHW'
        self.dtype = np.float32
        self.init_test_case()
        self.op_type = "trilinear_interp_v2"
        # NOTE(dev): some AsDispensible input is not used under imperative mode.
        # Skip check_dygraph while found them in Inputs.
        input_np = np.random.random(self.input_shape).astype(self.dtype)

        scale_w = 0
        scale_h = 0
        scale_d = 0
        if self.data_layout == "NCDHW":
            in_d = self.input_shape[2]
            in_h = self.input_shape[3]
            in_w = self.input_shape[4]
        else:
            in_d = self.input_shape[1]
            in_h = self.input_shape[2]
            in_w = self.input_shape[3]

        if self.scale:
            if isinstance(self.scale, (float, int)):
                if self.scale > 0:
                    scale_d = scale_h = scale_w = float(self.scale)
            if isinstance(self.scale, list) and len(self.scale) == 1:
                scale_d = scale_w = scale_h = self.scale[0]
            elif isinstance(self.scale, list) and len(self.scale) > 1:
                scale_w = self.scale[2]
                scale_h = self.scale[1]
                scale_d = self.scale[0]
            out_d = int(in_d * scale_d)
            out_h = int(in_h * scale_h)
            out_w = int(in_w * scale_w)
        else:
            out_d = self.out_d
            out_h = self.out_h
            out_w = self.out_w

        output_np = trilinear_interp_np(
            input_np,
            out_d,
            out_h,
            out_w,
            scale_d,
            scale_h,
            scale_w,
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
            # c++ end treat NCDHW the same way as NCHW
        if self.data_layout == 'NCDHW':
            data_layout = 'NCHW'
        else:
            data_layout = 'NHWC'
        self.attrs = {
            'out_d': self.out_d,
            'out_h': self.out_h,
            'out_w': self.out_w,
            'interp_method': self.interp_method,
            'align_corners': self.align_corners,
            'align_mode': self.align_mode,
            'data_layout': data_layout,
        }
        if self.scale:
            if isinstance(self.scale, (float, int)):
                if self.scale > 0:
                    self.scale = [self.scale]
            if isinstance(self.scale, list) and len(self.scale) == 1:
                self.scale = [self.scale[0], self.scale[0], self.scale[0]]
            self.attrs['scale'] = self.scale
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output(
            check_pir=True, check_symbol_infer=(self.out_size is None)
        )

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', in_place=True, check_pir=True)

    def init_test_case(self):
        create_test_case0(self)


class TestTrilinearInterpCase1(TestTrilinearInterpOp):
    def init_test_case(self):
        create_test_case1(self)


class TestTrilinearInterpCase2(TestTrilinearInterpOp):
    def init_test_case(self):
        create_test_case2(self)


class TestTrilinearInterpCase3(TestTrilinearInterpOp):
    def init_test_case(self):
        create_test_case3(self)


class TestTrilinearInterpCase4(TestTrilinearInterpOp):
    def init_test_case(self):
        create_test_case4(self)


class TestTrilinearInterpCase5(TestTrilinearInterpOp):
    def init_test_case(self):
        create_test_case5(self)


class TestTrilinearInterpCase6(TestTrilinearInterpOp):
    def init_test_case(self):
        create_test_case6(self)


class TestTrilinearInterpSame(TestTrilinearInterpOp):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [1, 1, 16, 8, 4]
        self.out_d = 16
        self.out_h = 8
        self.out_w = 4
        self.scale = []
        self.align_corners = True
        self.align_mode = 1


class TestTrilinearInterpSameHW(TestTrilinearInterpOp):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [1, 1, 16, 8, 4]
        self.out_d = 8
        self.out_h = 8
        self.out_w = 4
        self.scale = []
        self.align_corners = True
        self.align_mode = 1


class TestTrilinearInterpActualShape(TestTrilinearInterpOp):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [3, 2, 16, 8, 4]
        self.out_d = 64
        self.out_h = 32
        self.out_w = 16
        self.scale = []
        self.out_size = np.array([33, 19, 7]).astype("int32")
        self.align_corners = True
        self.align_mode = 1


class TestTrilinearInterpDatalayout(TestTrilinearInterpOp):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [2, 4, 4, 4, 3]
        self.out_d = 2
        self.out_h = 2
        self.out_w = 2
        self.scale = []
        self.out_size = np.array([3, 3, 3]).astype("int32")
        self.align_corners = True
        self.align_mode = 1
        self.data_layout = "NDHWC"


class TestTrilinearInterpOpFP16(TestTrilinearInterpOp):
    def test_check_output(self):
        self.check_output(
            check_pir=True, check_symbol_infer=(self.out_size is None)
        )

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', in_place=True, check_pir=True)

    def init_test_case(self):
        create_test_case0(self)
        self.dtype = np.float16


class TestTrilinearInterpCase1FP16(TestTrilinearInterpOpFP16):
    def init_test_case(self):
        create_test_case1(self)
        self.dtype = np.float16


class TestTrilinearInterpCase2FP16(TestTrilinearInterpOpFP16):
    def init_test_case(self):
        create_test_case2(self)
        self.dtype = np.float16


class TestTrilinearInterpCase3FP16(TestTrilinearInterpOpFP16):
    def init_test_case(self):
        create_test_case3(self)
        self.dtype = np.float16


class TestTrilinearInterpCase4FP16(TestTrilinearInterpOpFP16):
    def init_test_case(self):
        create_test_case4(self)
        self.dtype = np.float16


class TestTrilinearInterpCase5FP16(TestTrilinearInterpOpFP16):
    def init_test_case(self):
        create_test_case5(self)
        self.dtype = np.float16


class TestTrilinearInterpCase6FP16(TestTrilinearInterpOpFP16):
    def init_test_case(self):
        create_test_case6(self)
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support the bfloat16",
)
class TestNearestInterpOpBF16(OpTest):
    def setUp(self):
        self.python_api = trilinear_interp_test
        self.out_size = None
        self.actual_shape = None
        self.data_layout = 'NCDHW'
        self.init_test_case()
        self.op_type = "trilinear_interp_v2"
        # NOTE(dev): some AsDispensible input is not used under imperative mode.
        # Skip check_dygraph while found them in Inputs.
        self.dtype = np.uint16
        input_np = np.random.random(self.input_shape).astype("float32")

        scale_w = 0
        scale_h = 0
        scale_d = 0
        if self.data_layout == "NCDHW":
            in_d = self.input_shape[2]
            in_h = self.input_shape[3]
            in_w = self.input_shape[4]
        else:
            in_d = self.input_shape[1]
            in_h = self.input_shape[2]
            in_w = self.input_shape[3]

        if self.scale:
            if isinstance(self.scale, (float, int)):
                if self.scale > 0:
                    scale_d = scale_h = scale_w = float(self.scale)
            if isinstance(self.scale, list) and len(self.scale) == 1:
                scale_d = scale_w = scale_h = self.scale[0]
            elif isinstance(self.scale, list) and len(self.scale) > 1:
                scale_w = self.scale[2]
                scale_h = self.scale[1]
                scale_d = self.scale[0]
            out_d = int(in_d * scale_d)
            out_h = int(in_h * scale_h)
            out_w = int(in_w * scale_w)
        else:
            out_d = self.out_d
            out_h = self.out_h
            out_w = self.out_w

        output_np = trilinear_interp_np(
            input_np,
            out_d,
            out_h,
            out_w,
            scale_d,
            scale_h,
            scale_w,
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
        # c++ end treat NCDHW the same way as NCHW
        if self.data_layout == 'NCDHW':
            data_layout = 'NCHW'
        else:
            data_layout = 'NHWC'
        self.attrs = {
            'out_d': self.out_d,
            'out_h': self.out_h,
            'out_w': self.out_w,
            'interp_method': self.interp_method,
            'align_corners': self.align_corners,
            'align_mode': self.align_mode,
            'data_layout': data_layout,
        }
        if self.scale:
            if isinstance(self.scale, (float, int)):
                if self.scale > 0:
                    self.scale = [self.scale]
            if isinstance(self.scale, list) and len(self.scale) == 1:
                self.scale = [self.scale[0], self.scale[0], self.scale[0]]
            self.attrs['scale'] = self.scale
        self.outputs = {'Out': convert_float_to_uint16(output_np)}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', in_place=True, check_pir=True)

    def init_test_case(self):
        create_test_case0(self)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support the bfloat16",
)
class TestTrilinearInterpCase1BF16(TestNearestInterpOpBF16):
    def init_test_case(self):
        create_test_case1(self)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support the bfloat16",
)
class TestTrilinearInterpCase2BF16(TestNearestInterpOpBF16):
    def init_test_case(self):
        create_test_case2(self)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support the bfloat16",
)
class TestTrilinearInterpCase3BF16(TestNearestInterpOpBF16):
    def init_test_case(self):
        create_test_case3(self)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support the bfloat16",
)
class TestTrilinearInterpCase4BF16(TestNearestInterpOpBF16):
    def init_test_case(self):
        create_test_case4(self)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support the bfloat16",
)
class TestTrilinearInterpCase5BF16(TestNearestInterpOpBF16):
    def init_test_case(self):
        create_test_case5(self)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support the bfloat16",
)
class TestTrilinearInterpCase6BF16(TestNearestInterpOpBF16):
    def init_test_case(self):
        create_test_case6(self)


class TestTrilinearInterpOpUint8(OpTest):
    def setUp(self):
        self.python_api = trilinear_interp_test
        self.out_size = None
        self.actual_shape = None
        self.init_test_case()
        self.op_type = "trilinear_interp_v2"
        input_np = np.random.randint(
            low=0, high=256, size=self.input_shape
        ).astype("uint8")

        if self.scale:
            if isinstance(self.scale, (float, int)):
                if self.scale > 0:
                    scale_d = scale_h = scale_w = float(self.scale)
            if isinstance(self.scale, list) and len(self.scale) == 1:
                scale_d = scale_w = scale_h = self.scale[0]
            elif isinstance(self.scale, list) and len(self.scale) > 1:
                scale_w = self.scale[2]
                scale_h = self.scale[1]
                scale_d = self.scale[0]
            out_d = int(self.input_shape[2] * scale_d)
            out_h = int(self.input_shape[3] * scale_h)
            out_w = int(self.input_shape[4] * scale_w)
        else:
            out_d = self.out_d
            out_h = self.out_h
            out_w = self.out_w

        output_np = trilinear_interp_np(
            input_np,
            out_d,
            out_h,
            out_w,
            0,
            0,
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
            'out_d': self.out_d,
            'out_h': self.out_h,
            'out_w': self.out_w,
            'interp_method': self.interp_method,
            'align_corners': self.align_corners,
            'align_mode': self.align_mode,
        }
        if self.scale:
            if isinstance(self.scale, (float, int)):
                if self.scale > 0:
                    self.scale = [self.scale]
            if isinstance(self.scale, list) and len(self.scale) == 1:
                self.scale = [self.scale[0], self.scale[0], self.scale[0]]
            self.attrs['scale'] = self.scale
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output_with_place(
            place=core.CPUPlace(),
            atol=1,
            check_pir=True,
            check_symbol_infer=(self.out_size is None),
        )

    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [1, 3, 9, 6, 8]
        self.out_d = 13
        self.out_h = 10
        self.out_w = 9
        self.scale = []
        self.align_corners = True
        self.align_mode = 1


class TestTrilinearInterpCase1Uint8(TestTrilinearInterpOpUint8):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [2, 3, 16, 8, 4]
        self.out_d = 13
        self.out_h = 7
        self.out_w = 2
        self.scale = []
        self.align_corners = True
        self.align_mode = 1


class TestTrilinearInterpCase2Uint8(TestTrilinearInterpOpUint8):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [4, 1, 7, 8, 9]
        self.out_d = 3
        self.out_h = 5
        self.out_w = 13
        self.scale = []
        self.out_size = np.array([6, 15, 21]).astype("int32")
        self.align_corners = True
        self.align_mode = 1


class TestTrilinearInterpOtherMethod1(TestTrilinearInterpOp):
    def set_align_mode(self):
        self.align_corners = False
        self.align_mode = 1


class TestTrilinearInterpWithMethod2(TestTrilinearInterpOp):
    def set_align_mode(self):
        self.align_corners = False
        self.align_mode = 0


class TestTrilinearInterpWithMethod3(TestTrilinearInterpOp):
    def set_align_mode(self):
        self.align_corners = True
        self.align_mode = 0


class TestTrilinearInterpScale1(TestTrilinearInterpOp):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [2, 3, 5, 7, 9]
        self.out_d = 19
        self.out_h = 15
        self.out_w = 8
        self.scale = 2.0
        self.align_corners = True
        self.align_mode = 1


class TestTrilinearInterpScale2(TestTrilinearInterpOp):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [2, 3, 5, 7, 9]
        self.out_d = 30
        self.out_h = 20
        self.out_w = 25
        self.scale = 1.0
        self.align_corners = True
        self.align_mode = 1


class TestTrilinearInterpScale3(TestTrilinearInterpOp):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [2, 3, 5, 7, 9]
        self.out_d = 30
        self.out_h = 20
        self.out_w = 25
        self.scale = 1.5
        self.align_corners = True
        self.align_mode = 1


class TestTrilinearInterpZero(TestTrilinearInterpOp):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [2, 3, 5, 7, 11]
        self.out_d = 30
        self.out_h = 20
        self.out_w = 25
        self.scale = []
        self.align_corners = False
        self.align_mode = 0


class TestTrilinearInterpOp_attr_tensor(OpTest):
    def setUp(self):
        self.python_api = trilinear_interp_test
        self.out_size = None
        self.actual_shape = None
        self.init_test_case()
        self.op_type = "trilinear_interp_v2"
        self.shape_by_1Dtensor = False
        self.scale_by_1Dtensor = False
        self.attrs = {
            'interp_method': self.interp_method,
            'align_corners': self.align_corners,
            'align_mode': self.align_mode,
        }

        input_np = np.random.random(self.input_shape).astype("float32")
        self.inputs = {'X': input_np}

        if self.scale_by_1Dtensor:
            self.inputs['Scale'] = np.array([self.scale]).astype("float32")
        elif self.scale:
            if isinstance(self.scale, (float, int)):
                if self.scale > 0:
                    scale_d = scale_h = scale_w = float(self.scale)
            if isinstance(self.scale, list) and len(self.scale) == 1:
                scale_d = scale_w = scale_h = self.scale[0]
            elif isinstance(self.scale, list) and len(self.scale) > 1:
                scale_w = self.scale[2]
                scale_h = self.scale[1]
                scale_d = self.scale[0]
            out_d = int(self.input_shape[2] * scale_d)
            out_h = int(self.input_shape[3] * scale_h)
            out_w = int(self.input_shape[4] * scale_w)
        else:
            out_d = self.out_d
            out_h = self.out_h
            out_w = self.out_w

        if self.shape_by_1Dtensor:
            self.inputs['OutSize'] = self.out_size
        elif self.out_size is not None:
            size_tensor = []
            for index, ele in enumerate(self.out_size):
                size_tensor.append(
                    ("x" + str(index), np.ones(1).astype('int32') * ele)
                )
            self.inputs['SizeTensor'] = size_tensor

        self.attrs['out_d'] = self.out_d
        self.attrs['out_h'] = self.out_h
        self.attrs['out_w'] = self.out_w
        if self.scale:
            if isinstance(self.scale, (float, int)):
                if self.scale > 0:
                    self.scale = [self.scale]
            if isinstance(self.scale, list) and len(self.scale) == 1:
                self.scale = [self.scale[0], self.scale[0], self.scale[0]]
            self.attrs['scale'] = self.scale
        output_np = trilinear_interp_np(
            input_np,
            out_d,
            out_h,
            out_w,
            0,
            0,
            0,
            self.out_size,
            self.actual_shape,
            self.align_corners,
            self.align_mode,
        )
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output(
            check_pir=True, check_symbol_infer=(self.out_size is None)
        )

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', in_place=True, check_pir=True)

    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [2, 3, 4, 4, 4]
        self.out_d = 2
        self.out_h = 3
        self.out_w = 3
        self.scale = []
        self.out_size = [2, 3, 3]
        self.align_corners = True
        self.align_mode = 1


# out_size is a 1-D tensor
class TestTrilinearInterp_attr_tensor_Case1(TestTrilinearInterpOp_attr_tensor):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [3, 2, 9, 6, 8]
        self.out_d = 32
        self.out_h = 16
        self.out_w = 8
        self.scale = 0.3
        self.out_size = [12, 4, 4]
        self.align_corners = True
        self.align_mode = 1


# scale is a 1-D tensor
class TestTrilinearInterp_attr_tensor_Case2(TestTrilinearInterpOp_attr_tensor):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [2, 3, 8, 8, 4]
        self.out_d = 16
        self.out_h = 12
        self.out_w = 4
        self.scale = []
        self.out_size = [16, 4, 10]
        self.align_corners = True
        self.align_mode = 1
        self.shape_by_1Dtensor = True


# scale is a 1-D tensor
class TestTrilinearInterp_attr_tensor_Case3(TestTrilinearInterpOp_attr_tensor):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [2, 3, 8, 8, 4]
        self.out_d = 16
        self.out_h = 16
        self.out_w = 8
        self.scale = 2.0
        self.out_size = None
        self.align_corners = True
        self.align_mode = 1
        self.scale_by_1Dtensor = True


@unittest.skipIf(
    not base.core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestTrilinearInterpOpForFloat16(unittest.TestCase):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [2, 3, 4, 4, 4]
        self.out_size = np.array([3, 3, 3]).astype("int32")
        self.align_corners = True
        self.align_mode = 1
        self.data_layout = 'NCDHW'

    def check_main(self, x_np, dtype):
        paddle.disable_static()
        x_np = x_np.astype(dtype)
        x = paddle.to_tensor(x_np)
        x.stop_gradient = False
        y = interpolate(
            x,
            size=self.out_size.tolist(),
            mode=self.interp_method,
            align_corners=self.align_corners,
            align_mode=self.align_mode,
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
        np.testing.assert_allclose(x_g_np_1, x_g_np_2, rtol=1e-05)


@unittest.skipIf(
    not base.core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestTrilinearInterpDatalayoutForFloat16(TestTrilinearInterpOpForFloat16):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [2, 4, 4, 4, 3]
        self.out_size = np.array([3, 3, 3]).astype("int32")
        self.align_corners = True
        self.align_mode = 1
        self.data_layout = "NDHWC"


class TestTrilinearInterpOpAPI(unittest.TestCase):
    def test_case(self):
        import paddle

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        with base.dygraph.guard(place):
            input_data = np.random.random((2, 3, 6, 6, 6)).astype("float32")
            scale_np = np.array([2, 2, 2]).astype("int64")
            input_x = paddle.to_tensor(input_data)
            scale = paddle.to_tensor(scale_np)
            expect_res = trilinear_interp_np(
                input_data, out_d=12, out_h=12, out_w=12, align_corners=False
            )
            up_layer = paddle.nn.Upsample(
                scale_factor=scale, mode="trilinear", align_corners=False
            )
            out = up_layer(input_x)
            np.testing.assert_allclose(out.numpy(), expect_res, rtol=1e-05)


class TestTrilinearInterpOpAPI2(unittest.TestCase):
    def test_case(self):
        import paddle

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        with base.dygraph.guard(place):
            input_data = np.random.random((2, 3, 6, 6, 6)).astype("float32")
            scale_np = np.array([2, 2, 2]).astype("int64")
            input_x = paddle.to_tensor(input_data)
            scale = paddle.to_tensor(scale_np)
            expect_res = trilinear_interp_np(
                input_data, out_d=12, out_h=12, out_w=12, align_corners=False
            )
            out = interpolate(
                x=input_x,
                scale_factor=scale,
                mode="trilinear",
                align_corners=False,
            )
            np.testing.assert_allclose(out.numpy(), expect_res, rtol=1e-05)


if __name__ == "__main__":
    unittest.main()
