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

import sys

sys.path.append('..')
from op_test import OpTest
from test_strided_slice_op import strided_slice_native_forward
import numpy as np
import unittest
import paddle.fluid as fluid
import paddle

paddle.enable_static()


class TestStrideSliceOp(OpTest):
<<<<<<< HEAD
=======

>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
    def setUp(self):
        self.initTestCase()
        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True
        self.op_type = 'strided_slice'
        self.python_api = paddle.strided_slice
<<<<<<< HEAD
        self.output = strided_slice_native_forward(
            self.input, self.axes, self.starts, self.ends, self.strides
        )
=======
        self.output = strided_slice_native_forward(self.input, self.axes,
                                                   self.starts, self.ends,
                                                   self.strides)
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf

        self.inputs = {'Input': self.input}
        self.outputs = {'Out': self.output}
        self.attrs = {
            'axes': self.axes,
            'starts': self.starts,
            'ends': self.ends,
            'strides': self.strides,
<<<<<<< HEAD
            'infer_flags': self.infer_flags,
=======
            'infer_flags': self.infer_flags
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
        }

    def test_check_output(self):
        self.check_output_with_place(self.place, check_eager=False)

    def test_check_grad(self):
<<<<<<< HEAD
        self.check_grad_with_place(
            self.place, set(['Input']), 'Out', check_eager=False
        )
=======
        self.check_grad_with_place(self.place,
                                   set(['Input']),
                                   'Out',
                                   check_eager=False)
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf

    def initTestCase(self):
        self.input = np.random.rand(100).astype(np.float32)
        self.axes = [0]
        self.starts = [-4]
        self.ends = [-3]
        self.strides = [1]
        self.infer_flags = [1]


class TestStrideSliceOp1(TestStrideSliceOp):
<<<<<<< HEAD
=======

>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
    def initTestCase(self):
        self.input = np.random.rand(100).astype(np.float32)
        self.axes = [0]
        self.starts = [3]
        self.ends = [8]
        self.strides = [1]
        self.infer_flags = [1]


class TestStrideSliceOp2(TestStrideSliceOp):
<<<<<<< HEAD
=======

>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
    def initTestCase(self):
        self.input = np.random.rand(100).astype(np.float32)
        self.axes = [0]
        self.starts = [5]
        self.ends = [0]
        self.strides = [-1]
        self.infer_flags = [1]


class TestStrideSliceOp3(TestStrideSliceOp):
<<<<<<< HEAD
=======

>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
    def initTestCase(self):
        self.input = np.random.rand(100).astype(np.float32)
        self.axes = [0]
        self.starts = [-1]
        self.ends = [-3]
        self.strides = [-1]
        self.infer_flags = [1]


class TestStrideSliceOp4(TestStrideSliceOp):
<<<<<<< HEAD
=======

>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
    def initTestCase(self):
        self.input = np.random.rand(3, 4, 10).astype(np.float32)
        self.axes = [0, 1, 2]
        self.starts = [0, -1, 0]
        self.ends = [2, -3, 5]
        self.strides = [1, -1, 1]
        self.infer_flags = [1, 1, 1]


class TestStrideSliceOp5(TestStrideSliceOp):
<<<<<<< HEAD
=======

>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
    def initTestCase(self):
        self.input = np.random.rand(5, 5, 5).astype(np.float32)
        self.axes = [0, 1, 2]
        self.starts = [1, 0, 0]
        self.ends = [2, 1, 3]
        self.strides = [1, 1, 1]
        self.infer_flags = [1, 1, 1]


class TestStrideSliceOp6(TestStrideSliceOp):
<<<<<<< HEAD
=======

>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
    def initTestCase(self):
        self.input = np.random.rand(5, 5, 5).astype(np.float32)
        self.axes = [0, 1, 2]
        self.starts = [1, -1, 0]
        self.ends = [2, -3, 3]
        self.strides = [1, -1, 1]
        self.infer_flags = [1, 1, 1]


class TestStrideSliceOp7(TestStrideSliceOp):
<<<<<<< HEAD
=======

>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
    def initTestCase(self):
        self.input = np.random.rand(5, 5, 5).astype(np.float32)
        self.axes = [0, 1, 2]
        self.starts = [1, 0, 0]
        self.ends = [2, 2, 3]
        self.strides = [1, 1, 1]
        self.infer_flags = [1, 1, 1]


class TestStrideSliceOp8(TestStrideSliceOp):
<<<<<<< HEAD
=======

>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
    def initTestCase(self):
        self.input = np.random.rand(1, 100, 1).astype(np.float32)
        self.axes = [1]
        self.starts = [1]
        self.ends = [2]
        self.strides = [1]
        self.infer_flags = [1]


class TestStrideSliceOp9(TestStrideSliceOp):
<<<<<<< HEAD
=======

>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
    def initTestCase(self):
        self.input = np.random.rand(1, 100, 1).astype(np.float32)
        self.axes = [1]
        self.starts = [-1]
        self.ends = [-2]
        self.strides = [-1]
        self.infer_flags = [1]


class TestStrideSliceOp10(TestStrideSliceOp):
<<<<<<< HEAD
=======

>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
    def initTestCase(self):
        self.input = np.random.rand(10, 10).astype(np.float32)
        self.axes = [0, 1]
        self.starts = [1, 0]
        self.ends = [2, 2]
        self.strides = [1, 1]
        self.infer_flags = [1, 1]


class TestStrideSliceOp11(TestStrideSliceOp):
<<<<<<< HEAD
=======

>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
    def initTestCase(self):
        self.input = np.random.rand(3, 3, 3, 4).astype(np.float32)
        self.axes = [0, 1, 2, 3]
        self.starts = [1, 0, 0, 0]
        self.ends = [2, 2, 3, 4]
        self.strides = [1, 1, 1, 2]
        self.infer_flags = [1, 1, 1, 1]


class TestStrideSliceOp12(TestStrideSliceOp):
<<<<<<< HEAD
=======

>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
    def initTestCase(self):
        self.input = np.random.rand(3, 3, 3, 4, 5).astype(np.float32)
        self.axes = [0, 1, 2, 3, 4]
        self.starts = [1, 0, 0, 0, 0]
        self.ends = [2, 2, 3, 4, 4]
        self.strides = [1, 1, 1, 1, 1]
        self.infer_flags = [1, 1, 1, 1]


class TestStrideSliceOp13(TestStrideSliceOp):
<<<<<<< HEAD
=======

>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
    def initTestCase(self):
        self.input = np.random.rand(3, 3, 3, 6, 7, 8).astype(np.float32)
        self.axes = [0, 1, 2, 3, 4, 5]
        self.starts = [1, 0, 0, 0, 1, 2]
        self.ends = [2, 2, 3, 1, 2, 8]
        self.strides = [1, 1, 1, 1, 1, 2]
        self.infer_flags = [1, 1, 1, 1, 1]


class TestStrideSliceOp14(TestStrideSliceOp):
<<<<<<< HEAD
=======

>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
    def initTestCase(self):
        self.input = np.random.rand(4, 4, 4, 4).astype(np.float32)
        self.axes = [1, 2, 3]
        self.starts = [-5, 0, -7]
        self.ends = [-1, 2, 4]
        self.strides = [1, 1, 1]
        self.infer_flags = [1, 1, 1]


class TestStrideSliceOpBool(TestStrideSliceOp):
<<<<<<< HEAD
=======

>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
    def test_check_grad(self):
        pass


class TestStrideSliceOpBool1D(TestStrideSliceOpBool):
<<<<<<< HEAD
=======

>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
    def initTestCase(self):
        self.input = np.random.rand(100).astype("bool")
        self.axes = [0]
        self.starts = [3]
        self.ends = [8]
        self.strides = [1]
        self.infer_flags = [1]


class TestStrideSliceOpBool2D(TestStrideSliceOpBool):
<<<<<<< HEAD
=======

>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
    def initTestCase(self):
        self.input = np.random.rand(10, 10).astype("bool")
        self.axes = [0, 1]
        self.starts = [1, 0]
        self.ends = [2, 2]
        self.strides = [1, 1]
        self.infer_flags = [1, 1]


class TestStrideSliceOpBool3D(TestStrideSliceOpBool):
<<<<<<< HEAD
=======

>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
    def initTestCase(self):
        self.input = np.random.rand(3, 4, 10).astype("bool")
        self.axes = [0, 1, 2]
        self.starts = [0, -1, 0]
        self.ends = [2, -3, 5]
        self.strides = [1, -1, 1]
        self.infer_flags = [1, 1, 1]


class TestStrideSliceOpBool4D(TestStrideSliceOpBool):
<<<<<<< HEAD
=======

>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
    def initTestCase(self):
        self.input = np.random.rand(3, 3, 3, 4).astype("bool")
        self.axes = [0, 1, 2, 3]
        self.starts = [1, 0, 0, 0]
        self.ends = [2, 2, 3, 4]
        self.strides = [1, 1, 1, 2]
        self.infer_flags = [1, 1, 1, 1]


class TestStrideSliceOpBool5D(TestStrideSliceOpBool):
<<<<<<< HEAD
=======

>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
    def initTestCase(self):
        self.input = np.random.rand(3, 3, 3, 4, 5).astype("bool")
        self.axes = [0, 1, 2, 3, 4]
        self.starts = [1, 0, 0, 0, 0]
        self.ends = [2, 2, 3, 4, 4]
        self.strides = [1, 1, 1, 1, 1]
        self.infer_flags = [1, 1, 1, 1]


class TestStrideSliceOpBool6D(TestStrideSliceOpBool):
<<<<<<< HEAD
=======

>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
    def initTestCase(self):
        self.input = np.random.rand(3, 3, 3, 6, 7, 8).astype("bool")
        self.axes = [0, 1, 2, 3, 4, 5]
        self.starts = [1, 0, 0, 0, 1, 2]
        self.ends = [2, 2, 3, 1, 2, 8]
        self.strides = [1, 1, 1, 1, 1, 2]
        self.infer_flags = [1, 1, 1, 1, 1]


class TestStridedSliceOp_starts_ListTensor(OpTest):
<<<<<<< HEAD
=======

>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
    def setUp(self):
        self.op_type = "strided_slice"
        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True
        self.config()

        starts_tensor = []
        for index, ele in enumerate(self.starts):
<<<<<<< HEAD
            starts_tensor.append(
                ("x" + str(index), np.ones((1)).astype('int32') * ele)
            )
=======
            starts_tensor.append(("x" + str(index), np.ones(
                (1)).astype('int32') * ele))
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf

        self.inputs = {'Input': self.input, 'StartsTensorList': starts_tensor}
        self.outputs = {'Out': self.output}
        self.attrs = {
            'axes': self.axes,
            'starts': self.starts_infer,
            'ends': self.ends,
            'strides': self.strides,
<<<<<<< HEAD
            'infer_flags': self.infer_flags,
=======
            'infer_flags': self.infer_flags
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float32")
        self.starts = [1, 0, 2]
        self.ends = [3, 3, 4]
        self.axes = [0, 1, 2]
        self.strides = [1, 1, 1]
        self.infer_flags = [1, -1, 1]
<<<<<<< HEAD
        self.output = strided_slice_native_forward(
            self.input, self.axes, self.starts, self.ends, self.strides
        )
=======
        self.output = strided_slice_native_forward(self.input, self.axes,
                                                   self.starts, self.ends,
                                                   self.strides)
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf

        self.starts_infer = [1, 10, 2]

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
<<<<<<< HEAD
        self.check_grad_with_place(
            self.place, ['Input'], 'Out', max_relative_error=0.006
        )


class TestStridedSliceOp_ends_ListTensor(OpTest):
=======
        self.check_grad_with_place(self.place, ['Input'],
                                   'Out',
                                   max_relative_error=0.006)


class TestStridedSliceOp_ends_ListTensor(OpTest):

>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
    def setUp(self):
        self.op_type = "strided_slice"
        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True
        self.config()

        ends_tensor = []
        for index, ele in enumerate(self.ends):
<<<<<<< HEAD
            ends_tensor.append(
                ("x" + str(index), np.ones((1)).astype('int32') * ele)
            )
=======
            ends_tensor.append(("x" + str(index), np.ones(
                (1)).astype('int32') * ele))
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf

        self.inputs = {'Input': self.input, 'EndsTensorList': ends_tensor}
        self.outputs = {'Out': self.output}
        self.attrs = {
            'axes': self.axes,
            'starts': self.starts,
            'ends': self.ends_infer,
            'strides': self.strides,
<<<<<<< HEAD
            'infer_flags': self.infer_flags,
=======
            'infer_flags': self.infer_flags
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float32")
        self.starts = [1, 0, 0]
        self.ends = [3, 3, 4]
        self.axes = [0, 1, 2]
        self.strides = [1, 1, 2]
        self.infer_flags = [1, -1, 1]
<<<<<<< HEAD
        self.output = strided_slice_native_forward(
            self.input, self.axes, self.starts, self.ends, self.strides
        )
=======
        self.output = strided_slice_native_forward(self.input, self.axes,
                                                   self.starts, self.ends,
                                                   self.strides)
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf

        self.ends_infer = [3, 1, 4]

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
<<<<<<< HEAD
        self.check_grad_with_place(
            self.place, ['Input'], 'Out', max_relative_error=0.006
        )


class TestStridedSliceOp_starts_Tensor(OpTest):
=======
        self.check_grad_with_place(self.place, ['Input'],
                                   'Out',
                                   max_relative_error=0.006)


class TestStridedSliceOp_starts_Tensor(OpTest):

>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
    def setUp(self):
        self.op_type = "strided_slice"
        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True
        self.config()
        self.inputs = {
            'Input': self.input,
<<<<<<< HEAD
            "StartsTensor": np.array(self.starts, dtype="int32"),
=======
            "StartsTensor": np.array(self.starts, dtype="int32")
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
        }
        self.outputs = {'Out': self.output}
        self.attrs = {
            'axes': self.axes,
            #'starts': self.starts,
            'ends': self.ends,
            'strides': self.strides,
            'infer_flags': self.infer_flags,
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float32")
        self.starts = [1, 0, 2]
        self.ends = [2, 3, 4]
        self.axes = [0, 1, 2]
        self.strides = [1, 1, 1]
        self.infer_flags = [-1, -1, -1]
<<<<<<< HEAD
        self.output = strided_slice_native_forward(
            self.input, self.axes, self.starts, self.ends, self.strides
        )
=======
        self.output = strided_slice_native_forward(self.input, self.axes,
                                                   self.starts, self.ends,
                                                   self.strides)
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
<<<<<<< HEAD
        self.check_grad_with_place(
            self.place, ['Input'], 'Out', max_relative_error=0.006
        )


class TestStridedSliceOp_ends_Tensor(OpTest):
=======
        self.check_grad_with_place(self.place, ['Input'],
                                   'Out',
                                   max_relative_error=0.006)


class TestStridedSliceOp_ends_Tensor(OpTest):

>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
    def setUp(self):
        self.op_type = "strided_slice"
        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True
        self.config()
        self.inputs = {
            'Input': self.input,
<<<<<<< HEAD
            "EndsTensor": np.array(self.ends, dtype="int32"),
=======
            "EndsTensor": np.array(self.ends, dtype="int32")
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
        }
        self.outputs = {'Out': self.output}
        self.attrs = {
            'axes': self.axes,
            'starts': self.starts,
            #'ends': self.ends,
            'strides': self.strides,
            'infer_flags': self.infer_flags,
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float32")
        self.starts = [1, 0, 2]
        self.ends = [2, 3, 4]
        self.axes = [0, 1, 2]
        self.strides = [1, 1, 1]
        self.infer_flags = [-1, -1, -1]
<<<<<<< HEAD
        self.output = strided_slice_native_forward(
            self.input, self.axes, self.starts, self.ends, self.strides
        )
=======
        self.output = strided_slice_native_forward(self.input, self.axes,
                                                   self.starts, self.ends,
                                                   self.strides)
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
<<<<<<< HEAD
        self.check_grad_with_place(
            self.place, ['Input'], 'Out', max_relative_error=0.006
        )


class TestStridedSliceOp_listTensor_Tensor(OpTest):
=======
        self.check_grad_with_place(self.place, ['Input'],
                                   'Out',
                                   max_relative_error=0.006)


class TestStridedSliceOp_listTensor_Tensor(OpTest):

>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
    def setUp(self):
        self.config()
        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True
        ends_tensor = []
        for index, ele in enumerate(self.ends):
<<<<<<< HEAD
            ends_tensor.append(
                ("x" + str(index), np.ones((1)).astype('int32') * ele)
            )
=======
            ends_tensor.append(("x" + str(index), np.ones(
                (1)).astype('int32') * ele))
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
        self.op_type = "strided_slice"

        self.inputs = {
            'Input': self.input,
            "StartsTensor": np.array(self.starts, dtype="int32"),
<<<<<<< HEAD
            "EndsTensorList": ends_tensor,
=======
            "EndsTensorList": ends_tensor
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
        }
        self.outputs = {'Out': self.output}
        self.attrs = {
            'axes': self.axes,
            #'starts': self.starts,
            #'ends': self.ends,
            'strides': self.strides,
            'infer_flags': self.infer_flags,
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float32")
        self.starts = [1, 0, 2]
        self.ends = [2, 3, 4]
        self.axes = [0, 1, 2]
        self.strides = [1, 1, 1]
        self.infer_flags = [-1, -1, -1]
<<<<<<< HEAD
        self.output = strided_slice_native_forward(
            self.input, self.axes, self.starts, self.ends, self.strides
        )
=======
        self.output = strided_slice_native_forward(self.input, self.axes,
                                                   self.starts, self.ends,
                                                   self.strides)
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
<<<<<<< HEAD
        self.check_grad_with_place(
            self.place, ['Input'], 'Out', max_relative_error=0.006
        )


class TestStridedSliceOp_strides_Tensor(OpTest):
=======
        self.check_grad_with_place(self.place, ['Input'],
                                   'Out',
                                   max_relative_error=0.006)


class TestStridedSliceOp_strides_Tensor(OpTest):

>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
    def setUp(self):
        self.op_type = "strided_slice"
        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True
        self.config()
        self.inputs = {
            'Input': self.input,
<<<<<<< HEAD
            "StridesTensor": np.array(self.strides, dtype="int32"),
=======
            "StridesTensor": np.array(self.strides, dtype="int32")
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
        }
        self.outputs = {'Out': self.output}
        self.attrs = {
            'axes': self.axes,
            'starts': self.starts,
            'ends': self.ends,
            #'strides': self.strides,
            'infer_flags': self.infer_flags,
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float32")
        self.starts = [1, -1, 2]
        self.ends = [2, 0, 4]
        self.axes = [0, 1, 2]
        self.strides = [1, -1, 1]
        self.infer_flags = [-1, -1, -1]
<<<<<<< HEAD
        self.output = strided_slice_native_forward(
            self.input, self.axes, self.starts, self.ends, self.strides
        )
=======
        self.output = strided_slice_native_forward(self.input, self.axes,
                                                   self.starts, self.ends,
                                                   self.strides)
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
<<<<<<< HEAD
        self.check_grad_with_place(
            self.place, ['Input'], 'Out', max_relative_error=0.006
        )
=======
        self.check_grad_with_place(self.place, ['Input'],
                                   'Out',
                                   max_relative_error=0.006)
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf


# Test python API
class TestStridedSliceAPI(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
    def test_1(self):
        input = np.random.random([3, 4, 5, 6]).astype("float32")
        minus_1 = fluid.layers.fill_constant([1], "int32", -1)
        minus_3 = fluid.layers.fill_constant([1], "int32", -3)
<<<<<<< HEAD
        starts = fluid.layers.data(
            name='starts', shape=[3], dtype='int32', append_batch_size=False
        )
        ends = fluid.layers.data(
            name='ends', shape=[3], dtype='int32', append_batch_size=False
        )
        strides = fluid.layers.data(
            name='strides', shape=[3], dtype='int32', append_batch_size=False
        )

        x = fluid.layers.data(
            name="x",
            shape=[3, 4, 5, 6],
            append_batch_size=False,
            dtype="float32",
        )
        out_1 = paddle.strided_slice(
            x,
            axes=[0, 1, 2],
            starts=[-3, 0, 2],
            ends=[3, 100, -1],
            strides=[1, 1, 1],
        )
        out_2 = paddle.strided_slice(
            x,
            axes=[0, 1, 3],
            starts=[minus_3, 0, 2],
            ends=[3, 100, -1],
            strides=[1, 1, 1],
        )
        out_3 = paddle.strided_slice(
            x,
            axes=[0, 1, 3],
            starts=[minus_3, 0, 2],
            ends=[3, 100, minus_1],
            strides=[1, 1, 1],
        )
        out_4 = paddle.strided_slice(
            x, axes=[0, 1, 2], starts=starts, ends=ends, strides=strides
        )
=======
        starts = fluid.layers.data(name='starts',
                                   shape=[3],
                                   dtype='int32',
                                   append_batch_size=False)
        ends = fluid.layers.data(name='ends',
                                 shape=[3],
                                 dtype='int32',
                                 append_batch_size=False)
        strides = fluid.layers.data(name='strides',
                                    shape=[3],
                                    dtype='int32',
                                    append_batch_size=False)

        x = fluid.layers.data(name="x",
                              shape=[3, 4, 5, 6],
                              append_batch_size=False,
                              dtype="float32")
        out_1 = paddle.strided_slice(x,
                                     axes=[0, 1, 2],
                                     starts=[-3, 0, 2],
                                     ends=[3, 100, -1],
                                     strides=[1, 1, 1])
        out_2 = paddle.strided_slice(x,
                                     axes=[0, 1, 3],
                                     starts=[minus_3, 0, 2],
                                     ends=[3, 100, -1],
                                     strides=[1, 1, 1])
        out_3 = paddle.strided_slice(x,
                                     axes=[0, 1, 3],
                                     starts=[minus_3, 0, 2],
                                     ends=[3, 100, minus_1],
                                     strides=[1, 1, 1])
        out_4 = paddle.strided_slice(x,
                                     axes=[0, 1, 2],
                                     starts=starts,
                                     ends=ends,
                                     strides=strides)
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf

        out_5 = x[-3:3, 0:100:2, -1:2:-1]
        out_6 = x[minus_3:3:1, 0:100:2, :, minus_1:2:minus_1]
        out_7 = x[minus_1, 0:100:2, :, -1:2:-1]

        exe = fluid.Executor(place=fluid.MLUPlace(0))
        res_1, res_2, res_3, res_4, res_5, res_6, res_7 = exe.run(
            fluid.default_main_program(),
            feed={
                "x": input,
                'starts': np.array([-3, 0, 2]).astype("int32"),
                'ends': np.array([3, 2147483648, -1]).astype("int64"),
<<<<<<< HEAD
                'strides': np.array([1, 1, 1]).astype("int32"),
            },
            fetch_list=[out_1, out_2, out_3, out_4, out_5, out_6, out_7],
        )
=======
                'strides': np.array([1, 1, 1]).astype("int32")
            },
            fetch_list=[out_1, out_2, out_3, out_4, out_5, out_6, out_7])
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
        assert np.array_equal(res_1, input[-3:3, 0:100, 2:-1, :])
        assert np.array_equal(res_2, input[-3:3, 0:100, :, 2:-1])
        assert np.array_equal(res_3, input[-3:3, 0:100, :, 2:-1])
        assert np.array_equal(res_4, input[-3:3, 0:100, 2:-1, :])
        assert np.array_equal(res_5, input[-3:3, 0:100:2, -1:2:-1, :])
        assert np.array_equal(res_6, input[-3:3, 0:100:2, :, -1:2:-1])
        assert np.array_equal(res_7, input[-1, 0:100:2, :, -1:2:-1])

    def test_dygraph_op(self):
        x = paddle.zeros(shape=[3, 4, 5, 6], dtype="float32")
        axes = [1, 2, 3]
        starts = [-3, 0, 2]
        ends = [3, 2, 4]
        strides_1 = [1, 1, 1]
<<<<<<< HEAD
        sliced_1 = paddle.strided_slice(
            x, axes=axes, starts=starts, ends=ends, strides=strides_1
        )
=======
        sliced_1 = paddle.strided_slice(x,
                                        axes=axes,
                                        starts=starts,
                                        ends=ends,
                                        strides=strides_1)
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
        assert sliced_1.shape == (3, 2, 2, 2)


if __name__ == "__main__":
    unittest.main()
