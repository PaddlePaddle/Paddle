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
import sys

sys.path.append("..")
from op_test import OpTest
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.nn.functional import interpolate
import paddle

from test_bilinear_interp_v2_op import bilinear_interp_np

paddle.enable_static()


class TestBilinearInterpOp(OpTest):

    def set_npu(self):
        self.__class__.use_npu = True
        self.place = paddle.NPUPlace(0)

    def setUp(self):
        self.set_npu()
        self.out_size = None
        self.actual_shape = None
        self.data_layout = 'NCHW'
        self.init_test_case()
        self.op_type = "bilinear_interp_v2"
        input_np = np.random.random(self.input_shape).astype(self.dtype)

        if self.data_layout == "NCHW":
            in_h = self.input_shape[2]
            in_w = self.input_shape[3]
        else:
            in_h = self.input_shape[1]
            in_w = self.input_shape[2]
        scale_h = 0
        scale_w = 0
        if self.scale:
            if isinstance(self.scale, float) or isinstance(self.scale, int):
                if self.scale > 0.:
                    scale_h = scale_w = float(self.scale)
            if isinstance(self.scale, list) and len(self.scale) == 1:
                scale_w = scale_h = self.scale[0]
            elif isinstance(self.scale, list) and len(self.scale) > 1:
                scale_w = self.scale[1]
                scale_h = self.scale[0]
            out_h = int(in_h * scale_h)
            out_w = int(in_w * scale_w)
        else:
            out_h = self.out_h
            out_w = self.out_w

        output_np = bilinear_interp_np(input_np, out_h, out_w, scale_w, scale_h,
                                       self.out_size, self.actual_shape,
                                       self.align_corners, self.align_mode,
                                       self.data_layout)

        self.inputs = {'X': input_np}
        if self.out_size is not None:
            self.inputs['OutSize'] = self.out_size
        if self.actual_shape is not None:
            self.inputs['OutSize'] = self.actual_shape

        self.attrs = {
            'out_h': self.out_h,
            'out_w': self.out_w,
            'interp_method': self.interp_method,
            'align_corners': self.align_corners,
            'align_mode': self.align_mode,
            'data_layout': self.data_layout
        }
        if self.scale:
            if isinstance(self.scale, float) or isinstance(self.scale, int):
                if self.scale > 0.:
                    self.scale = [self.scale]
            if isinstance(self.scale, list) and len(self.scale) == 1:
                self.scale = [self.scale[0], self.scale[0]]
            self.attrs['scale'] = self.scale
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=self.atol)

    def test_check_grad(self):
        self.__class__.exist_check_grad = True
        if self.dtype == 'float16':
            return
        self.max_relative_error = 0.005
        inputs_to_check = ['X']
        output_names = ['Out']
        no_grad_set = set()
        cpu_place = fluid.CPUPlace()
        cpu_grads = self._get_gradient(inputs_to_check, cpu_place, output_names,
                                       no_grad_set)
        npu_grads = self._get_gradient(inputs_to_check, self.place,
                                       output_names, no_grad_set)
        self._assert_is_close(cpu_grads, npu_grads, inputs_to_check,
                              self.max_relative_error,
                              "Gradient Check between places")

    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [2, 3, 5, 7]
        self.out_h = 60
        self.out_w = 25
        self.scale = 1.5
        self.align_corners = False
        self.align_mode = 1
        self.dtype = 'float32'
        self.atol = 1e-5


class TestBilinearInterpCaseFP16(TestBilinearInterpOp):

    def init_test_case(self):
        super(TestBilinearInterpCaseFP16, self).init_test_case()
        self.dtype = 'float16'
        self.atol = 1e-2


class TestBilinearInterpCase1(TestBilinearInterpOp):

    def init_test_case(self):
        super(TestBilinearInterpCase1, self).init_test_case()
        self.input_shape = [4, 1, 7, 8]
        self.out_h = 1
        self.out_w = 1
        self.scale = 0.


class TestBilinearInterpCase2(TestBilinearInterpOp):

    def init_test_case(self):
        super(TestBilinearInterpCase2, self).init_test_case()
        self.input_shape = [3, 3, 9, 6]
        self.out_h = 12
        self.out_w = 12
        self.scale = 0.


class TestBilinearInterpCase3(TestBilinearInterpOp):

    def init_test_case(self):
        super(TestBilinearInterpCase3, self).init_test_case()
        self.input_shape = [1, 1, 32, 64]
        self.out_h = 64
        self.out_w = 32
        self.scale = 0.


class TestBilinearInterpCase4(TestBilinearInterpOp):

    def init_test_case(self):
        super(TestBilinearInterpCase4, self).init_test_case()
        self.input_shape = [4, 1, 7, 8]
        self.out_h = 1
        self.out_w = 1
        self.scale = 0.
        self.out_size = np.array([2, 2]).astype("int32")


class TestBilinearInterpCase5(TestBilinearInterpOp):

    def init_test_case(self):
        super(TestBilinearInterpCase5, self).init_test_case()
        self.input_shape = [3, 3, 9, 6]
        self.out_h = 12
        self.out_w = 12
        self.scale = 0.
        self.out_size = np.array([11, 11]).astype("int32")


class TestBilinearInterpCase6(TestBilinearInterpOp):

    def init_test_case(self):
        super(TestBilinearInterpCase6, self).init_test_case()
        self.input_shape = [1, 1, 32, 64]
        self.out_h = 64
        self.out_w = 32
        self.scale = 0.
        self.out_size = np.array([65, 33]).astype("int32")


class TestBilinearInterpCase7(TestBilinearInterpOp):

    def init_test_case(self):
        super(TestBilinearInterpCase7, self).init_test_case()
        self.input_shape = [1, 1, 32, 64]
        self.out_h = 64
        self.out_w = 32
        self.scale = [2.0, 0.5]


class TestBilinearInterpSame(TestBilinearInterpOp):

    def init_test_case(self):
        super(TestBilinearInterpSame, self).init_test_case()
        self.input_shape = [2, 3, 32, 64]
        self.out_h = 32
        self.out_w = 64
        self.scale = 0.


class TestBilinearInterpActualShape(TestBilinearInterpOp):

    def init_test_case(self):
        super(TestBilinearInterpActualShape, self).init_test_case()
        self.input_shape = [3, 2, 32, 16]
        self.out_h = 64
        self.out_w = 32
        self.scale = 0.
        self.out_size = np.array([66, 40]).astype("int32")


class TestBilinearInterpDataLayout(TestBilinearInterpOp):

    def init_test_case(self):
        super(TestBilinearInterpDataLayout, self).init_test_case()
        self.input_shape = [2, 5, 5, 3]
        self.out_h = 2
        self.out_w = 2
        self.scale = 0.
        self.out_size = np.array([3, 3]).astype("int32")
        self.data_layout = "NHWC"


class TestBilinearInterpOtherMethod1(TestBilinearInterpOp):

    def set_align_mode(self):
        self.align_corners = False
        self.align_mode = 1


class TestBilinearInterpWithMethod2(TestBilinearInterpOp):

    def set_align_mode(self):
        self.align_corners = False
        self.align_mode = 0


class TestBilinearInterpWithMethod3(TestBilinearInterpOp):

    def set_align_mode(self):
        self.align_corners = True
        self.align_mode = 0


class TestBilinearInterpScale1(TestBilinearInterpOp):

    def init_test_case(self):
        super(TestBilinearInterpScale1, self).init_test_case()
        self.input_shape = [2, 3, 5, 7]
        self.out_h = 60
        self.out_w = 25
        self.scale = 2.


class TestBilinearInterpScale2(TestBilinearInterpOp):

    def init_test_case(self):
        super(TestBilinearInterpScale2, self).init_test_case()
        self.input_shape = [2, 3, 5, 7]
        self.out_h = 60
        self.out_w = 25
        self.scale = 1.


class TestBilinearInterpZero(TestBilinearInterpOp):

    def init_test_case(self):
        super(TestBilinearInterpZero, self).init_test_case()
        self.input_shape = [2, 3, 5, 7]
        self.out_h = 60
        self.out_w = 25
        self.scale = 0.2
        self.align_mode = 0


if __name__ == "__main__":
    unittest.main()
