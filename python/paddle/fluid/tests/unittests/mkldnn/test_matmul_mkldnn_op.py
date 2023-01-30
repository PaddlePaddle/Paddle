# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

<<<<<<< HEAD
import os
import unittest

import numpy as np

=======
from __future__ import print_function

import unittest, os
import numpy as np
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
from paddle.fluid.tests.unittests.op_test import OpTest, skip_check_grad_ci


class TestDnnlMatMulOp(OpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def generate_data(self):
        self.x = np.random.random((25, 2, 2)).astype("float32")
        self.y = np.random.random((25, 2, 2)).astype("float32")
        self.alpha = 1.0
        self.out = self.alpha * np.matmul(self.x, self.y)

    def set_attributes(self):
        self.alpha = self.alpha if hasattr(self, 'alpha') else 1.0
        self.attrs = {'alpha': self.alpha}

    def setUp(self):
        # Set max isa, otherwise fails on SKX and earlier
        os.environ["DNNL_MAX_CPU_ISA"] = "AVX"
        self.op_type = "matmul"
        self._cpu_only = True
        self.use_mkldnn = True
        self.generate_data()
        self.set_attributes()
        self.attrs['use_mkldnn'] = True

        self.inputs = {'X': self.x, 'Y': self.y}
        self.outputs = {'Out': self.out}

    def test_check_output(self):
        self.check_output()


class TestDnnlMatMulWithGradOp(TestDnnlMatMulOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_check_grad(self):
        self.check_grad(['X', 'Y'], 'Out', max_relative_error=1e-2)


class TestDnnlMatMulOpMixedDims1(TestDnnlMatMulWithGradOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def generate_data(self):
        self.x = np.random.random((17, 2, 3)).astype("float32")
        self.y = np.random.random((3, 4)).astype("float32")
        self.out = np.matmul(self.x, self.y)


class TestDnnlMatMulOpMixedDimsYWiderTransposeY(TestDnnlMatMulWithGradOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def generate_data(self):
        self.x = np.random.random((8, 2, 3)).astype("float32")
        self.y = np.random.random((4, 3)).astype("float32")
        self.out = np.matmul(self.x, np.transpose(self.y))

    def set_attributes(self):
        self.attrs = {'transpose_Y': True}


class TestDnnlMatMulOpMixedDimsYWiderTransposeX(TestDnnlMatMulWithGradOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def generate_data(self):
        self.x = np.random.random((8, 3, 2)).astype("float32")
        self.y = np.random.random((3, 4)).astype("float32")
        self.out = np.matmul(np.transpose(self.x, (0, 2, 1)), self.y)

    def set_attributes(self):
        self.attrs = {'transpose_X': True}


class TestDnnlMatMulOpMixedDimsXWiderTransposeXY(TestDnnlMatMulWithGradOp):
<<<<<<< HEAD
    def generate_data(self):
        self.x = np.random.random((8, 3, 2)).astype("float32")
        self.y = np.random.random((4, 3)).astype("float32")
        self.out = np.matmul(
            np.transpose(self.x, (0, 2, 1)), np.transpose(self.y)
        )
=======

    def generate_data(self):
        self.x = np.random.random((8, 3, 2)).astype("float32")
        self.y = np.random.random((4, 3)).astype("float32")
        self.out = np.matmul(np.transpose(self.x, (0, 2, 1)),
                             np.transpose(self.y))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def set_attributes(self):
        self.attrs = {'transpose_X': True, 'transpose_Y': True}


class TestDnnlMatMulOpMixedDimsYWiderTransposeXY(TestDnnlMatMulWithGradOp):
<<<<<<< HEAD
    def generate_data(self):
        self.x = np.random.random((3, 2)).astype("float32")
        self.y = np.random.random((8, 4, 3)).astype("float32")
        self.out = np.matmul(
            np.transpose(self.x), np.transpose(self.y, (0, 2, 1))
        )
=======

    def generate_data(self):
        self.x = np.random.random((3, 2)).astype("float32")
        self.y = np.random.random((8, 4, 3)).astype("float32")
        self.out = np.matmul(np.transpose(self.x),
                             np.transpose(self.y, (0, 2, 1)))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def set_attributes(self):
        self.attrs = {'transpose_X': True, 'transpose_Y': True}


class TestDnnlMatMulOpMixedDimsXWiderTransposeX(TestDnnlMatMulWithGradOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def generate_data(self):
        self.x = np.random.random((5, 4)).astype("float32")
        self.y = np.random.random((8, 5, 4)).astype("float32")
        self.out = np.matmul(np.transpose(self.x), self.y)

    def set_attributes(self):
        self.attrs = {'transpose_X': True}


class TestDnnlMatMulOpVectorMultiply(TestDnnlMatMulWithGradOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def generate_data(self):
        self.x = np.random.random((5)).astype("float32")
        self.y = np.random.random((5)).astype("float32")
        self.out = np.matmul(self.x, self.y)


class TestDnnlMatMulOpVectorMultiplyTranspose(TestDnnlMatMulWithGradOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def generate_data(self):
        self.x = np.random.random((5)).astype("float32")
        x_resized = np.copy(self.x)
        x_resized = np.expand_dims(x_resized, 1)
        self.y = np.random.random((6)).astype("float32")
        y_resized = np.copy(self.y)
        y_resized = np.expand_dims(y_resized, 0)
        self.out = np.matmul(x_resized, y_resized)

    def set_attributes(self):
        self.attrs = {'transpose_Y': True, 'transpose_X': True}


class TestDnnlMatMulOpMixedDims2(TestDnnlMatMulWithGradOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def generate_data(self):
        self.x = np.random.random((2, 3)).astype("float32")
        self.y = np.random.random((17, 3, 4)).astype("float32")
        self.out = np.matmul(self.x, self.y)


class TestDnnlMatMulOpAlpha(TestDnnlMatMulWithGradOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def generate_data(self):
        self.x = np.random.random((17, 2, 3)).astype("float32")
        self.y = np.random.random((17, 3, 2)).astype("float32")
        self.alpha = 2.0
        self.out = self.alpha * np.matmul(self.x, self.y)


class TestDnnlMatMulOp2D(TestDnnlMatMulWithGradOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def generate_data(self):
        self.x = np.random.random((12, 9)).astype("float32")
        self.y = np.random.random((9, 12)).astype("float32")
        self.out = np.matmul(self.x, self.y)


class TestDnnlMatMulOpTransposeX(TestDnnlMatMulWithGradOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def generate_data(self):
        self.x = np.random.random((12, 9)).astype("float32")
        self.y = np.random.random((12, 9)).astype("float32")
        self.out = np.matmul(np.transpose(self.x), self.y)

    def set_attributes(self):
        self.attrs = {'transpose_X': True}


class TestDnnlMatMulOpTransposeY(TestDnnlMatMulWithGradOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def generate_data(self):
        self.x = np.random.random((12, 9)).astype("float32")
        self.y = np.random.random((12, 9)).astype("float32")
        self.out = np.matmul(self.x, np.transpose(self.y))

    def set_attributes(self):
        self.attrs = {'transpose_Y': True}


class TestDnnlMatMulOpTransposeY3D(TestDnnlMatMulWithGradOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def generate_data(self):
        self.x = np.random.random((17, 3, 2)).astype("float32")
        self.y = np.random.random((17, 3, 2)).astype("float32")
        self.out = np.matmul(self.x, np.transpose(self.y, (0, 2, 1)))

    def set_attributes(self):
        self.attrs = {'transpose_Y': True}


class TestDnnlMatMulOpInt8NoScales(TestDnnlMatMulOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def generate_data(self):
        self.x = np.random.random((12, 9)).astype("int8")
        self.y = np.random.random((9, 12)).astype("int8")
        self.out = np.matmul(self.x, self.y)


class TestDnnlMatMulOpInt8(TestDnnlMatMulOp):
    # Due to limitation in int8 matmul implementation
    # on older platforms (BDW, SKX) we needed to reduce
    # range from [-127, 127] to [-63, 63]
    def quantize(self, tensor):
<<<<<<< HEAD
        scale = 63.0 / np.abs(np.amax(tensor))
=======
        scale = 63. / np.abs(np.amax(tensor))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        quantized = np.round(scale * tensor).astype("int8")
        return scale, quantized

    def generate_data(self):
        x_float = np.random.random((12, 9)).astype("float32")
        self.x_scale, self.x = self.quantize(x_float)

        y_float = np.random.random((9, 12)).astype("float32")
        self.y_scale, self.y = self.quantize(y_float)

        out_float = np.matmul(x_float, y_float)
        self.out_scale, self.out = self.quantize(out_float)

    def set_attributes(self):
        self.attrs = {
            'Scale_x': self.x_scale,
            'Scale_y': self.y_scale,
            'Scale_out': self.out_scale,
        }

    def test_check_output(self):
        int_atol = 1
        self.check_output(atol=int_atol)


class TestDnnlMatMulOpInt8ForceFP32(TestDnnlMatMulOpInt8):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def generate_data(self):
        x_float = np.random.random((12, 9)).astype("float32")
        self.x_scale, self.x = self.quantize(x_float)

        y_float = np.random.random((9, 12)).astype("float32")
        self.y_scale, self.y = self.quantize(y_float)

        out_float = np.matmul(x_float, y_float)
        self.out = out_float

    def set_attributes(self):
        self.attrs = {
            'Scale_x': self.x_scale,
            'Scale_y': self.y_scale,
<<<<<<< HEAD
            'force_fp32_output': True,
=======
            'force_fp32_output': True
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }


class TestDnnlMatMulOpInt8ForceFP32BasicScales(TestDnnlMatMulOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def generate_data(self):
        self.x = np.random.randint(0, 3, (12, 9)).astype("int8")
        self.y = np.random.randint(0, 3, (9, 12)).astype("int8")
        self.out = np.matmul(self.x, self.y).astype("float32")

    def set_attributes(self):
        self.attrs = {'force_fp32_output': True}


@skip_check_grad_ci(reason="DNNL's MatMul doesn't implement grad kernel.")
class TestReshapeTransposeMatMulOp(OpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_data_type(self):
        self.data_type_ = 'float32'

    def generate_data(self):
<<<<<<< HEAD
        self.x = (
            np.random.random([2, 128, 768])
            .astype("float32")
            .reshape([2, 128, 12, 64])
            .transpose([0, 2, 1, 3])
        )
        self.y = (
            np.random.random([2, 128, 768])
            .astype("float32")
            .reshape([2, 128, 12, 64])
            .transpose([0, 2, 1, 3])
        )
=======
        self.x = np.random.random([2, 128, 768]).astype("float32").reshape(
            [2, 128, 12, 64]).transpose([0, 2, 1, 3])
        self.y = np.random.random([2, 128, 768]).astype("float32").reshape(
            [2, 128, 12, 64]).transpose([0, 2, 1, 3])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.out = np.matmul(self.x, self.y.transpose([0, 1, 3, 2]))
        self.fused_reshape_X = []
        self.fused_transpose_X = []
        self.fused_reshape_Y = []
        self.fused_transpose_Y = []

    def set_op_type_and_transpose_y_name(self):
        self.op_type = "matmul"
        self.transpose_y_name = "transpose_Y"

    def setUp(self):
        self.set_op_type_and_transpose_y_name()
        self._cpu_only = True
        self.use_mkldnn = True
        self.transpose_y = True
        self.init_data_type()
        self.generate_data()

        self.inputs = {'X': self.x, 'Y': self.y}
        self.attrs = {
            'use_mkldnn': self.use_mkldnn,
<<<<<<< HEAD
            self.transpose_y_name: self.transpose_y,
=======
            self.transpose_y_name: self.transpose_y
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }
        if len(self.fused_transpose_X) > 0:
            self.attrs['fused_transpose_X'] = self.fused_transpose_X
        if len(self.fused_transpose_Y) > 0:
            self.attrs['fused_transpose_Y'] = self.fused_transpose_Y
        if len(self.fused_reshape_X) > 0:
            self.attrs['fused_reshape_X'] = self.fused_reshape_X
        if len(self.fused_reshape_Y) > 0:
            self.attrs['fused_reshape_Y'] = self.fused_reshape_Y

        self.outputs = {'Out': self.out}

    def test_check_output(self):
        self.check_output()


class TestReshapeTransposeMatMulOp4DXFloat(TestReshapeTransposeMatMulOp):
<<<<<<< HEAD
    def generate_data(self):
        self.x = np.random.random([2, 128, 768]).astype("float32")
        self.y = (
            np.random.random([2, 128, 768])
            .astype("float32")
            .reshape([2, 128, 12, 64])
            .transpose([0, 2, 1, 3])
        )
=======

    def generate_data(self):
        self.x = np.random.random([2, 128, 768]).astype("float32")
        self.y = np.random.random([2, 128, 768]).astype("float32").reshape(
            [2, 128, 12, 64]).transpose([0, 2, 1, 3])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.fused_transpose_X = [0, 2, 1, 3]
        self.fused_reshape_X = [0, 0, 12, 64]
        self.fused_transpose_Y = []
        self.fused_reshape_Y = []
        self.out = np.matmul(
            self.x.reshape([2, 128, 12, 64]).transpose([0, 2, 1, 3]),
<<<<<<< HEAD
            self.y.transpose([0, 1, 3, 2]),
        )


class TestReshapeTransposeMatMulOp4DXInt8(TestReshapeTransposeMatMulOp4DXFloat):
=======
            self.y.transpose([0, 1, 3, 2]))


class TestReshapeTransposeMatMulOp4DXInt8(TestReshapeTransposeMatMulOp4DXFloat):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_data_type(self):
        self.data_type_ = 'int8'


class TestReshapeTransposeMatMulOp4DYFloat(TestReshapeTransposeMatMulOp):
<<<<<<< HEAD
    def generate_data(self):
        self.x = (
            np.random.random([2, 128, 768])
            .astype("float32")
            .reshape([2, 128, 12, 64])
            .transpose([0, 2, 1, 3])
        )
=======

    def generate_data(self):
        self.x = np.random.random([2, 128, 768]).astype("float32").reshape(
            [2, 128, 12, 64]).transpose([0, 2, 1, 3])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.y = np.random.random([2, 128, 768]).astype("float32")
        self.fused_transpose_X = []
        self.fused_reshape_X = []
        self.fused_transpose_Y = [0, 2, 1, 3]
        self.fused_reshape_Y = [0, 0, 12, 64]
        self.out = np.matmul(
<<<<<<< HEAD
            self.x, self.y.reshape([2, 128, 12, 64]).transpose([0, 2, 3, 1])
        )


class TestReshapeTransposeMatMulOp4DYInt8(TestReshapeTransposeMatMulOp4DYFloat):
=======
            self.x,
            self.y.reshape([2, 128, 12, 64]).transpose([0, 2, 3, 1]))


class TestReshapeTransposeMatMulOp4DYInt8(TestReshapeTransposeMatMulOp4DYFloat):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_data_type(self):
        self.data_type_ = 'int8'


class TestReshapeTransposeMatMulOp4DXYFloat(TestReshapeTransposeMatMulOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def generate_data(self):
        self.x = np.random.random([2, 128, 768]).astype("float32")
        self.y = np.random.random([2, 128, 768]).astype("float32")
        self.fused_transpose_X = [0, 2, 1, 3]
        self.fused_reshape_X = [0, 0, 12, 64]
        self.fused_transpose_Y = [0, 2, 1, 3]
        self.fused_reshape_Y = [0, 0, 12, 64]
        self.out = np.matmul(
            self.x.reshape([2, 128, 12, 64]).transpose([0, 2, 1, 3]),
<<<<<<< HEAD
            self.y.reshape([2, 128, 12, 64]).transpose([0, 2, 3, 1]),
        )


class TestReshapeTransposeMatMulOp4DXYInt8(
    TestReshapeTransposeMatMulOp4DXYFloat
):
=======
            self.y.reshape([2, 128, 12, 64]).transpose([0, 2, 3, 1]))


class TestReshapeTransposeMatMulOp4DXYInt8(TestReshapeTransposeMatMulOp4DXYFloat
                                           ):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_data_type(self):
        self.data_type_ = 'int8'


class TestReshapeTransposeMatMulOp2DXFloat(TestReshapeTransposeMatMulOp):
<<<<<<< HEAD
    def generate_data(self):
        self.x = np.random.random([2, 5, 10]).astype("float32")
        self.y = (
            np.random.random([2, 5, 10])
            .astype("float32")
            .reshape([10, 10])
            .transpose([1, 0])
        )
=======

    def generate_data(self):
        self.x = np.random.random([2, 5, 10]).astype("float32")
        self.y = np.random.random([2, 5, 10]).astype("float32").reshape(
            [10, 10]).transpose([1, 0])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.fused_transpose_X = [1, 0]
        self.fused_reshape_X = [10, 10]
        self.fused_transpose_Y = []
        self.fused_reshape_Y = []
        self.out = np.matmul(
<<<<<<< HEAD
            self.x.reshape([10, 10]).transpose([1, 0]), self.y.transpose([1, 0])
        )


class TestReshapeTransposeMatMulOp2DXInt8(TestReshapeTransposeMatMulOp2DXFloat):
=======
            self.x.reshape([10, 10]).transpose([1, 0]), self.y.transpose([1,
                                                                          0]))


class TestReshapeTransposeMatMulOp2DXInt8(TestReshapeTransposeMatMulOp2DXFloat):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_data_type(self):
        self.data_type_ = 'int8'


class TestReshapeTransposeMatMulOp2DYFloat(TestReshapeTransposeMatMulOp):
<<<<<<< HEAD
    def generate_data(self):
        self.x = (
            np.random.random([2, 5, 10])
            .astype("float32")
            .reshape([10, 10])
            .transpose([1, 0])
        )
=======

    def generate_data(self):
        self.x = np.random.random([2, 5, 10]).astype("float32").reshape(
            [10, 10]).transpose([1, 0])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.y = np.random.random([2, 5, 10]).astype("float32")
        self.fused_transpose_X = []
        self.fused_reshape_X = []
        self.fused_transpose_Y = [1, 0]
        self.fused_reshape_Y = [10, 10]
        self.out = np.matmul(self.x, self.y.reshape([10, 10]))


class TestReshapeTransposeMatMulOp2DYInt8(TestReshapeTransposeMatMulOp2DYFloat):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_data_type(self):
        self.data_type_ = 'int8'


class TestReshapeTransposeMatMulOp3DXFloat(TestReshapeTransposeMatMulOp):
<<<<<<< HEAD
    def generate_data(self):
        self.x = np.random.random([2, 2, 5, 5]).astype("float32")
        self.y = (
            np.random.random([2, 2, 5, 5])
            .astype("float32")
            .reshape([2, 10, 5])
            .transpose([0, 2, 1])
        )
=======

    def generate_data(self):
        self.x = np.random.random([2, 2, 5, 5]).astype("float32")
        self.y = np.random.random([2, 2, 5, 5]).astype("float32").reshape(
            [2, 10, 5]).transpose([0, 2, 1])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.fused_transpose_X = [0, 2, 1]
        self.fused_reshape_X = [2, 10, 5]
        self.fused_transpose_Y = []
        self.fused_reshape_Y = []
        self.out = np.matmul(
            self.x.reshape([2, 10, 5]).transpose(0, 2, 1),
<<<<<<< HEAD
            self.y.transpose(0, 2, 1),
        )


class TestReshapeTransposeMatMulOp3DXInt8(TestReshapeTransposeMatMulOp3DXFloat):
=======
            self.y.transpose(0, 2, 1))


class TestReshapeTransposeMatMulOp3DXInt8(TestReshapeTransposeMatMulOp3DXFloat):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_data_type(self):
        self.data_type_ = 'int8'


class TestReshapeTransposeMatMulOp3DYFloat(TestReshapeTransposeMatMulOp):
<<<<<<< HEAD
    def generate_data(self):
        self.x = (
            np.random.random([2, 2, 5, 5])
            .astype(self.data_type_)
            .reshape([2, 10, 5])
            .transpose([0, 2, 1])
        )
=======

    def generate_data(self):
        self.x = np.random.random([2, 2, 5, 5]).astype(self.data_type_).reshape(
            [2, 10, 5]).transpose([0, 2, 1])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.y = np.random.random([2, 2, 5, 5]).astype(self.data_type_)
        self.fused_transpose_X = []
        self.fused_reshape_X = []
        self.fused_transpose_Y = [0, 2, 1]
        self.fused_reshape_Y = [2, 10, 5]
        self.out = np.matmul(self.x, self.y.reshape([2, 10, 5]))


class TestReshapeTransposeMatMulOp3DYInt8(TestReshapeTransposeMatMulOp3DYFloat):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_data_type(self):
        self.data_type_ = 'int8'


@skip_check_grad_ci(reason="Tests inference only optimization.")
class TestMatMulOpTransposeReshapeEmptyFloat(OpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_data_type(self):
        self.data_type_ = np.float32

    def generate_data(self):
        self.bs = 1
        self.x = np.random.random([self.bs, 128, 128]).astype(self.data_type_)
        self.y = np.random.random([self.bs, 128, 64]).astype(self.data_type_)

    def init_params_and_out(self):
        self.transpose_out = []
        self.reshape_out = []
        self.out = np.matmul(self.x, self.y)

    def set_op_type(self):
        self.op_type = "matmul"

    def setUp(self):
        self.set_op_type()
        self._cpu_only = True
        self.use_mkldnn = True
        self.init_data_type()
        self.generate_data()
        self.init_params_and_out()

        self.inputs = {'X': self.x, 'Y': self.y}
        self.attrs = {'use_mkldnn': self.use_mkldnn}

        if len(self.reshape_out) > 0:
            self.attrs['fused_reshape_Out'] = self.reshape_out
        if len(self.transpose_out) > 0:
            self.attrs['fused_transpose_Out'] = self.transpose_out

        self.inputs = {'X': self.x, 'Y': self.y}
        self.outputs = {'Out': self.out}

    def test_check_output(self):
        self.check_output()

    def check_raise_error(self, msg):
        try:
            self.check_output()
        except Exception as e:
            if msg in str(e):
                raise AttributeError
            else:
                print(e)


class TestMatMulOpTransposeReshapeIntEmptyInt(
<<<<<<< HEAD
    TestMatMulOpTransposeReshapeEmptyFloat
):
=======
        TestMatMulOpTransposeReshapeEmptyFloat):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_data_type(self):
        self.data_type_ = np.int8


class TestMatMulOpTransposeReshapeBasicFloat(
<<<<<<< HEAD
    TestMatMulOpTransposeReshapeEmptyFloat
):
    def generate_data(self):
        self.bs = 8
        self.x = np.random.random([self.bs, 12, 128, 128]).astype(
            self.data_type_
        )
        self.y = np.random.random([self.bs, 12, 128, 64]).astype(
            self.data_type_
        )
=======
        TestMatMulOpTransposeReshapeEmptyFloat):

    def generate_data(self):
        self.bs = 8
        self.x = np.random.random([self.bs, 12, 128,
                                   128]).astype(self.data_type_)
        self.y = np.random.random([self.bs, 12, 128,
                                   64]).astype(self.data_type_)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def init_params_and_out(self):
        self.transpose_out = [0, 2, 1, 3]
        self.reshape_out = [0, 0, self.x.shape[1] * self.y.shape[-1]]
<<<<<<< HEAD
        self.out = (
            np.matmul(self.x, self.y)
            .transpose([0, 2, 1, 3])
            .reshape([self.bs, -1, self.x.shape[1] * self.y.shape[-1]])
        )


class TestMatMulOpTransposeReshapeBasicInt(
    TestMatMulOpTransposeReshapeBasicFloat
):
=======
        self.out = np.matmul(self.x, self.y).transpose([0, 2, 1, 3]).reshape(
            [self.bs, -1, self.x.shape[1] * self.y.shape[-1]])


class TestMatMulOpTransposeReshapeBasicInt(
        TestMatMulOpTransposeReshapeBasicFloat):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_data_type(self):
        self.data_type_ = np.int8


class TestMatMulOpTransposeReshapeOtherDimFloat(
<<<<<<< HEAD
    TestMatMulOpTransposeReshapeBasicFloat
):
=======
        TestMatMulOpTransposeReshapeBasicFloat):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def generate_data(self):
        self.bs = 11
        self.x = np.random.random([self.bs, 12, 14, 18]).astype(self.data_type_)
        self.y = np.random.random([self.bs, 12, 18, 13]).astype(self.data_type_)


class TestMatMulOpTransposeReshapeOtherDimInt(
<<<<<<< HEAD
    TestMatMulOpTransposeReshapeOtherDimFloat
):
=======
        TestMatMulOpTransposeReshapeOtherDimFloat):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_data_type(self):
        self.data_type_ = np.int8


if __name__ == "__main__":
    from paddle import enable_static
<<<<<<< HEAD

=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    enable_static()
    unittest.main()
