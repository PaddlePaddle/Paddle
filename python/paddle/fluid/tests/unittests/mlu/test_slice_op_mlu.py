#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid.core as core
import sys

sys.path.append('..')
from op_test import OpTest
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle

paddle.enable_static()


# Situation 1: starts(list, no tensor), ends(list, no tensor)
# 1.1 without attr(decrease)
class TestSliceOp(OpTest):

    def setUp(self):
        self.op_type = "slice"
        self.set_mlu()
        self.config()
        self.inputs = {'Input': self.input}
        self.outputs = {'Out': self.out}
        self.attrs = {
            'axes': self.axes,
            'starts': self.starts,
            'ends': self.ends,
            'infer_flags': self.infer_flags
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float32")
        self.starts = [1, 0, 2]
        self.ends = [3, 3, 4]
        self.axes = [0, 1, 2]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[1:3, 0:3, 2:4, :]

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        self.check_grad_with_place(self.place, ['Input'],
                                   'Out',
                                   max_relative_error=0.006)

    def set_mlu(self):
        self.__class__.use_mlu = True
        self.place = paddle.MLUPlace(0)


class TestCase1(TestSliceOp):

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float32")
        self.starts = [-3, 0, 2]
        self.ends = [3, 100, -1]
        self.axes = [0, 1, 2]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[-3:3, 0:100, 2:-1, :]


class TestCase2(TestSliceOp):

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float32")
        self.starts = [-3, 0, 2]
        self.ends = [3, 100, -1]
        self.axes = [0, 1, 3]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[-3:3, 0:100, :, 2:-1]


# 1.2 with attr(decrease)
class TestSliceOp_decs_dim(OpTest):

    def setUp(self):
        self.op_type = "slice"
        self.set_mlu()
        self.config()
        self.inputs = {'Input': self.input}
        self.outputs = {'Out': self.out}
        self.attrs = {
            'axes': self.axes,
            'starts': self.starts,
            'ends': self.ends,
            'infer_flags': self.infer_flags,
            'decrease_axis': self.decrease_axis,
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float32")
        self.starts = [1, 0, 2]
        self.ends = [2, 3, 4]
        self.axes = [0, 1, 2]
        self.decrease_axis = [0]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[1, 0:3, 2:4, :]

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        self.check_grad_with_place(self.place, ['Input'],
                                   'Out',
                                   max_relative_error=0.006)

    def set_mlu(self):
        self.__class__.use_mlu = True
        self.place = paddle.MLUPlace(0)


class TestSliceOp_decs_dim_2(TestSliceOp_decs_dim):

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float32")
        self.starts = [1, 0, 2]
        self.ends = [2, 1, 4]
        self.axes = [0, 1, 2]
        self.decrease_axis = [0, 1]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[1, 0, 2:4, :]


class TestSliceOp_decs_dim_3(TestSliceOp_decs_dim):

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float32")
        self.starts = [-1, 0, 2]
        self.ends = [1000000, 1, 4]
        self.axes = [0, 1, 2]
        self.decrease_axis = [0, 1]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[-1, 0, 2:4, :]


class TestSliceOp_decs_dim_4(TestSliceOp_decs_dim):

    def config(self):
        self.input = np.random.random([3, 4, 5, 7]).astype("float32")
        self.starts = [0, 1, 2, 3]
        self.ends = [1, 2, 3, 4]
        self.axes = [0, 1, 2, 3]
        self.decrease_axis = [0, 1, 2, 3]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[0, 1, 2, 3:4]


class TestSliceOp_decs_dim_5(TestSliceOp_decs_dim):

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float32")
        self.starts = [-1]
        self.ends = [1000000]
        self.axes = [3]
        self.decrease_axis = [3]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[:, :, :, -1]


class TestSliceOp_decs_dim_6(TestSliceOp_decs_dim):

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float32")
        self.starts = [0, 1, 2, 3]
        self.ends = [1, 2, 3, 4]
        self.axes = [0, 1, 2, 3]
        self.decrease_axis = [0, 1, 2, 3]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[0, 1, 2, 3:4]


# Situation 2: starts(list, have tensor), ends(list, no tensor)
# without attr(decrease)
class TestSliceOp_starts_ListTensor(OpTest):

    def setUp(self):
        self.op_type = "slice"
        self.set_mlu()
        self.config()

        starts_tensor = []
        for index, ele in enumerate(self.starts):
            starts_tensor.append(("x" + str(index), np.ones(
                (1)).astype('int64') * ele))

        self.inputs = {'Input': self.input, 'StartsTensorList': starts_tensor}
        self.outputs = {'Out': self.out}
        self.attrs = {
            'axes': self.axes,
            'starts': self.starts_infer,
            'ends': self.ends,
            'infer_flags': self.infer_flags
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float32")
        self.starts = [1, 0, 2]
        self.ends = [3, 3, 4]
        self.axes = [0, 1, 2]
        self.infer_flags = [-1, 1, -1]
        self.out = self.input[1:3, 0:3, 2:4, :]

        self.starts_infer = [-1, 0, -1]

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        self.check_grad_with_place(self.place, ['Input'],
                                   'Out',
                                   max_relative_error=0.006)

    def set_mlu(self):
        self.__class__.use_mlu = True
        self.place = paddle.MLUPlace(0)


# Situation 2: starts(list, have tensor), ends(list, no tensor)
#  with attr(decrease)
class TestSliceOp_decs_dim_starts_ListTensor(OpTest):

    def setUp(self):
        self.op_type = "slice"
        self.set_mlu()
        self.config()

        starts_tensor = []
        for index, ele in enumerate(self.starts):
            starts_tensor.append(("x" + str(index), np.ones(
                (1)).astype('int32') * ele))

        self.inputs = {'Input': self.input, 'StartsTensorList': starts_tensor}

        self.outputs = {'Out': self.out}
        self.attrs = {
            'axes': self.axes,
            'starts': self.starts_infer,
            'ends': self.ends,
            'infer_flags': self.infer_flags,
            'decrease_axis': self.decrease_axis,
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float32")
        self.starts = [1, 0, 2]
        self.ends = [2, 3, 4]
        self.axes = [0, 1, 2]
        self.decrease_axis = [0]
        self.infer_flags = [1, -1, 1]
        self.out = self.input[1, 0:3, 2:4, :]

        self.starts_infer = [1, -1, 2]

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        self.check_grad_with_place(self.place, ['Input'],
                                   'Out',
                                   max_relative_error=0.006)

    def set_mlu(self):
        self.__class__.use_mlu = True
        self.place = paddle.MLUPlace(0)


class TestSliceOp_decs_dim_5_starts_ListTensor(
        TestSliceOp_decs_dim_starts_ListTensor):

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float32")
        self.starts = [-1]
        self.ends = [1000000]
        self.axes = [3]
        self.decrease_axis = [3]
        self.infer_flags = [-1]
        self.out = self.input[:, :, :, -1]

        self.starts_infer = [-1]


# Situation 3: starts(tensor), ends(list, no tensor)
# with attr(decrease)
class TestSliceOp_decs_dim_starts_OneTensor(OpTest):

    def setUp(self):
        self.op_type = "slice"
        self.__class__.use_mlu = True
        self.place = paddle.MLUPlace(0)
        self.config()
        self.inputs = {
            'Input': self.input,
            "StartsTensor": np.array(self.starts, dtype="int32")
        }
        self.outputs = {'Out': self.out}
        self.attrs = {
            'axes': self.axes,
            #'starts': self.starts,
            'ends': self.ends,
            'infer_flags': self.infer_flags,
            'decrease_axis': self.decrease_axis,
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float32")
        self.starts = [1, 0, 2]
        self.ends = [2, 3, 4]
        self.axes = [0, 1, 2]
        self.decrease_axis = [0]
        self.infer_flags = [-1, -1, -1]
        self.out = self.input[1, 0:3, 2:4, :]

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        self.check_grad_with_place(self.place, ['Input'],
                                   'Out',
                                   max_relative_error=0.006)


# Situation 4: starts(tensor), ends(tensor)
# without attr(decrease)
class TestSliceOp_starts_OneTensor_ends_OneTensor(OpTest):

    def setUp(self):
        self.op_type = "slice"
        self.__class__.use_mlu = True
        self.place = paddle.MLUPlace(0)
        self.config()

        self.inputs = {
            'Input': self.input,
            "StartsTensor": np.array(self.starts, dtype="int64"),
            "EndsTensor": np.array(self.ends, dtype="int32")
        }
        self.outputs = {'Out': self.out}
        self.attrs = {
            'axes': self.axes,
            #'starts': self.starts,
            #'ends': self.ends_infer,
            'infer_flags': self.infer_flags
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float32")
        self.starts = [1, 0, 2]
        self.ends = [3, 3, 4]
        self.axes = [0, 1, 2]
        self.infer_flags = [-1, -1, -1]
        self.out = self.input[1:3, 0:3, 2:4, :]

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        self.check_grad_with_place(self.place, ['Input'],
                                   'Out',
                                   max_relative_error=0.006)


# Situation 5: starts(tensor), ends(tensor)
#  with attr(decrease)
class TestSliceOp_decs_dim_starts_and_ends_OneTensor(OpTest):

    def setUp(self):
        self.op_type = "slice"
        self.__class__.use_mlu = True
        self.place = paddle.MLUPlace(0)
        self.config()
        self.inputs = {
            'Input': self.input,
            "StartsTensor": np.array(self.starts, dtype="int32"),
            "EndsTensor": np.array(self.ends, dtype="int32")
        }
        self.outputs = {'Out': self.out}
        self.attrs = {
            'axes': self.axes,
            #'starts': self.starts,
            #'ends': self.ends,
            'infer_flags': self.infer_flags,
            'decrease_axis': self.decrease_axis,
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float32")
        self.starts = [1, 0, 2]
        self.ends = [2, 1, 4]
        self.axes = [0, 1, 2]
        self.decrease_axis = [0, 1]
        self.infer_flags = [-1, -1, -1]
        self.out = self.input[1, 0, 2:4, :]

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        self.check_grad_with_place(self.place, ['Input'],
                                   'Out',
                                   max_relative_error=0.006)


# Situation 6: starts(tensor), ends(list, have tensor)
# without attr(decrease)
class TestSliceOp_starts_OneTensor_ends_ListTensor(OpTest):

    def setUp(self):
        self.op_type = "slice"
        self.__class__.use_mlu = True
        self.place = paddle.MLUPlace(0)
        self.config()

        ends_tensor = []
        for index, ele in enumerate(self.ends):
            ends_tensor.append(("y" + str(index), np.ones(
                (1)).astype('int32') * ele))

        self.inputs = {
            'Input': self.input,
            "StartsTensor": np.array(self.starts, dtype="int32"),
            'EndsTensorList': ends_tensor
        }
        self.outputs = {'Out': self.out}
        self.attrs = {
            'axes': self.axes,
            #'starts': self.starts,
            'ends': self.ends_infer,
            'infer_flags': self.infer_flags
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float32")
        self.starts = [1, 0, 2]
        self.ends = [3, 3, 4]
        self.axes = [0, 1, 2]
        self.infer_flags = [-1, -1, -1]
        self.out = self.input[1:3, 0:3, 2:4, :]

        self.ends_infer = [-1, 3, 4]

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        self.check_grad_with_place(self.place, ['Input'],
                                   'Out',
                                   max_relative_error=0.006)


# Test float16
class TestFP16(OpTest):

    def setUp(self):
        self.op_type = "slice"
        self.__class__.use_mlu = True
        self.place = paddle.MLUPlace(0)
        self.config()
        self.inputs = {'Input': self.input}
        self.outputs = {'Out': self.out}
        self.attrs = {
            'axes': self.axes,
            'starts': self.starts,
            'ends': self.ends,
            'infer_flags': self.infer_flags
        }

    def config(self):
        self.dtype = "float16"
        self.input = np.random.random([3, 4, 5, 6]).astype(self.dtype)
        self.starts = [-3, 0, 2]
        self.ends = [3, 100, -1]
        self.axes = [0, 1, 3]
        self.out = self.input[-3:3, 0:100, :, 2:-1]
        self.infer_flags = [1, 1, 1]

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-5)

    def test_check_grad_normal(self):
        self.check_grad_with_place(self.place, ['Input'],
                                   'Out',
                                   max_relative_error=0.006)


class TestFP16_2(OpTest):

    def setUp(self):
        self.op_type = "slice"
        self.__class__.use_mlu = True
        self.place = paddle.MLUPlace(0)
        self.config()
        self.inputs = {'Input': self.input}
        self.outputs = {'Out': self.out}
        self.attrs = {
            'axes': self.axes,
            'starts': self.starts,
            'ends': self.ends,
            'infer_flags': self.infer_flags
        }

    def config(self):
        self.dtype = "float16"
        self.input = np.random.random([3, 4, 10]).astype(self.dtype)
        self.starts = [0]
        self.ends = [1]
        self.axes = [1]
        self.out = self.input[:, 0:1, :]
        self.infer_flags = [1]

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-5)

    def test_check_grad_normal(self):
        self.check_grad_with_place(self.place, ['Input'],
                                   'Out',
                                   max_relative_error=0.006,
                                   numeric_grad_delta=0.5)


class TestSliceApiWithTensor(unittest.TestCase):

    def test_starts_ends_is_tensor(self):
        with paddle.fluid.dygraph.guard():
            a = paddle.rand(shape=[4, 5, 6], dtype='float32')
            axes = [0, 1, 2]
            starts = [-3, 0, 2]
            ends = [3, 2, 4]
            a_1 = paddle.slice(a,
                               axes=axes,
                               starts=paddle.to_tensor(starts, dtype='int32'),
                               ends=paddle.to_tensor(ends, dtype='int32'))
            a_2 = paddle.slice(a, axes=axes, starts=starts, ends=ends)

            np.testing.assert_allclose(a_1.numpy(), a_2.numpy())

    def test_bool_tensor(self):
        with paddle.fluid.dygraph.guard():
            array = (np.arange(60).reshape([3, 4, 5]) % 3).astype('bool')
            tt = paddle.to_tensor(array)
            tt.stop_gradient = False

            starts = [0, 1, 2]
            ends = [3, 5, 4]
            axes = [0, 1, 2]

            y_paddle = paddle.slice(tt, axes, starts, ends)
            y_np = tt[0:3, 1:5, 2:4]

            self.assertTrue(paddle.bool == y_paddle.dtype)
            np.testing.assert_array_equal(y_paddle.numpy(), y_np)


class TestImperativeVarBaseGetItem(unittest.TestCase):

    def test_getitem_with_long(self):
        with fluid.dygraph.guard():
            data = np.random.random((2, 80, 16128)).astype('float32')
            var = fluid.dygraph.to_variable(data)
            sliced = var[:, 10:, :var.shape[1]]  # var.shape[1] is 80L here
            self.assertEqual(sliced.shape, [2, 70, 80])

            sliced = var[:, var.shape[0]:, var.shape[0]:var.shape[1]]
            self.assertEqual(sliced.shape, [2, 78, 78])

    def test_getitem_with_float(self):

        def test_float_in_slice_item():
            with fluid.dygraph.guard():
                data = np.random.random((2, 80, 16128)).astype('float32')
                var = fluid.dygraph.to_variable(data)
                sliced = var[:, 1.1:, :var.shape[1]]

        self.assertRaises(Exception, test_float_in_slice_item)

        def test_float_in_index():
            with fluid.dygraph.guard():
                data = np.random.random((2, 80, 16128)).astype('float32')
                var = fluid.dygraph.to_variable(data)
                sliced = var[1.1]

        self.assertRaises(Exception, test_float_in_index)


class TestInferShape(unittest.TestCase):

    def test(self):
        x = paddle.ones(shape=[3, 4, 5])
        x.desc.set_shape([3, -1, 5])
        self.assertEqual(x.shape, (3, -1, 5))

        out0 = paddle.slice(x, axes=[1], starts=[0], ends=[3])
        self.assertEqual(out0.shape, (3, 3, 5))

    def test_axis_less_than_zero(self):

        # Using paddle.disable_static will make other unittests fail.
        with fluid.dygraph.guard():
            x_arr = np.arange(0, 24, dtype=np.float32).reshape([2, 3, 4])
            x = paddle.to_tensor(x_arr)

            pp_slice = paddle.slice(x, [
                100,
            ], [0], [1])
            np_slice = x_arr[:, :, 0:1]
            np.testing.assert_allclose(pp_slice, np_slice)

            pp_slice = paddle.slice(x, (-100, ), [0], [1])
            np_slice = x_arr[0:1]
            np.testing.assert_allclose(pp_slice, np_slice)

            x_arr = np.array([], dtype=np.float32)
            x = paddle.to_tensor(np.reshape(x_arr, (0, 0, 0)))

            starts = paddle.to_tensor(
                np.reshape(np.array([], dtype=np.int32), (0, )))
            ends = paddle.to_tensor(
                np.reshape(np.array([], dtype=np.int32), (0, )))

            with self.assertRaises(ValueError):
                paddle.slice(x, [-1000000], starts, ends)

            with self.assertRaises(ValueError):
                paddle.slice(x, [1000000], starts, ends)

            with self.assertRaises(ValueError):
                paddle.slice(x, [], starts, ends)

            with self.assertRaises(ValueError):
                paddle.slice(x, 0, starts, ends)


if __name__ == '__main__':
    unittest.main()
