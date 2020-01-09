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

from __future__ import print_function

import unittest
import numpy as np
import paddle.fluid.core as core
from op_test import OpTest
import paddle.fluid as fluid


# Situation 1: starts(list, no tensor), ends(list, no tensor)
# 1.1 without attr(decrease)
class TestSliceOp(OpTest):
    def setUp(self):
        self.op_type = "slice"
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
        self.input = np.random.random([3, 4, 5, 6]).astype("float64")
        self.starts = [1, 0, 2]
        self.ends = [3, 3, 4]
        self.axes = [0, 1, 2]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[1:3, 0:3, 2:4, :]

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['Input'], 'Out', max_relative_error=0.006)


class TestCase1(TestSliceOp):
    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float64")
        self.starts = [-3, 0, 2]
        self.ends = [3, 100, -1]
        self.axes = [0, 1, 2]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[-3:3, 0:100, 2:-1, :]


class TestCase2(TestSliceOp):
    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float64")
        self.starts = [-3, 0, 2]
        self.ends = [3, 100, -1]
        self.axes = [0, 1, 3]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[-3:3, 0:100, :, 2:-1]


# 1.2 with attr(decrease)
class TestSliceOp_decs_dim(OpTest):
    def setUp(self):
        self.op_type = "slice"
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
        self.input = np.random.random([3, 4, 5, 6]).astype("float64")
        self.starts = [1, 0, 2]
        self.ends = [2, 3, 4]
        self.axes = [0, 1, 2]
        self.decrease_axis = [0]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[1, 0:3, 2:4, :]

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['Input'], 'Out', max_relative_error=0.006)


class TestSliceOp_decs_dim_2(TestSliceOp_decs_dim):
    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float64")
        self.starts = [1, 0, 2]
        self.ends = [2, 1, 4]
        self.axes = [0, 1, 2]
        self.decrease_axis = [0, 1]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[1, 0, 2:4, :]


class TestSliceOp_decs_dim_3(TestSliceOp_decs_dim):
    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float64")
        self.starts = [-1, 0, 2]
        self.ends = [1000000, 1, 4]
        self.axes = [0, 1, 2]
        self.decrease_axis = [0, 1]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[-1, 0, 2:4, :]


class TestSliceOp_decs_dim_4(TestSliceOp_decs_dim):
    def config(self):
        self.input = np.random.random([3, 4, 5, 7]).astype("float64")
        self.starts = [0, 1, 2, 3]
        self.ends = [1, 2, 3, 4]
        self.axes = [0, 1, 2, 3]
        self.decrease_axis = [0, 1, 2, 3]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[0, 1, 2, 3:4]


class TestSliceOp_decs_dim_5(TestSliceOp_decs_dim):
    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float64")
        self.starts = [-1]
        self.ends = [1000000]
        self.axes = [3]
        self.decrease_axis = [3]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[:, :, :, -1]


class TestSliceOp_decs_dim_6(TestSliceOp_decs_dim):
    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float64")
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
            'infer_flags': self.infer_flags
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float64")
        self.starts = [1, 0, 2]
        self.ends = [3, 3, 4]
        self.axes = [0, 1, 2]
        self.infer_flags = [-1, 1, -1]
        self.out = self.input[1:3, 0:3, 2:4, :]

        self.starts_infer = [-1, 0, -1]

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['Input'], 'Out', max_relative_error=0.006)


# Situation 2: starts(list, have tensor), ends(list, no tensor)
#  with attr(decrease)
class TestSliceOp_decs_dim_starts_ListTensor(OpTest):
    def setUp(self):
        self.op_type = "slice"
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
        self.input = np.random.random([3, 4, 5, 6]).astype("float64")
        self.starts = [1, 0, 2]
        self.ends = [2, 3, 4]
        self.axes = [0, 1, 2]
        self.decrease_axis = [0]
        self.infer_flags = [1, -1, 1]
        self.out = self.input[1, 0:3, 2:4, :]

        self.starts_infer = [1, -1, 2]

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['Input'], 'Out', max_relative_error=0.006)


class TestSliceOp_decs_dim_5_starts_ListTensor(
        TestSliceOp_decs_dim_starts_ListTensor):
    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float64")
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
        self.config()
        self.inputs = {
            'Input': self.input,
            "StartsTensor": np.array(
                self.starts, dtype="int32")
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
        self.input = np.random.random([3, 4, 5, 6]).astype("float64")
        self.starts = [1, 0, 2]
        self.ends = [2, 3, 4]
        self.axes = [0, 1, 2]
        self.decrease_axis = [0]
        self.infer_flags = [-1, -1, -1]
        self.out = self.input[1, 0:3, 2:4, :]

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['Input'], 'Out', max_relative_error=0.006)


# Situation 4: starts(tensor), ends(tensor)
#  without attr(decrease)
class TestSliceOp_starts_OneTensor_ends_OneTensor(OpTest):
    def setUp(self):
        self.op_type = "slice"
        self.config()

        self.inputs = {
            'Input': self.input,
            "StartsTensor": np.array(
                self.starts, dtype="int32"),
            "EndsTensor": np.array(
                self.ends, dtype="int32")
        }
        self.outputs = {'Out': self.out}
        self.attrs = {
            'axes': self.axes,
            #'starts': self.starts,
            #'ends': self.ends_infer,
            'infer_flags': self.infer_flags
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float64")
        self.starts = [1, 0, 2]
        self.ends = [3, 3, 4]
        self.axes = [0, 1, 2]
        self.infer_flags = [-1, -1, -1]
        self.out = self.input[1:3, 0:3, 2:4, :]

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['Input'], 'Out', max_relative_error=0.006)


# Situation 5: starts(tensor), ends(tensor)
#  with attr(decrease)
class TestSliceOp_decs_dim_starts_and_ends_OneTensor(OpTest):
    def setUp(self):
        self.op_type = "slice"
        self.config()
        self.inputs = {
            'Input': self.input,
            "StartsTensor": np.array(
                self.starts, dtype="int32"),
            "EndsTensor": np.array(
                self.ends, dtype="int32")
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
        self.input = np.random.random([3, 4, 5, 6]).astype("float64")
        self.starts = [1, 0, 2]
        self.ends = [2, 1, 4]
        self.axes = [0, 1, 2]
        self.decrease_axis = [0, 1]
        self.infer_flags = [-1, -1, -1]
        self.out = self.input[1, 0, 2:4, :]

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['Input'], 'Out', max_relative_error=0.006)


# Situation 6: starts(tensor), ends(list, have tensor)
# without attr(decrease)
class TestSliceOp_starts_OneTensor_ends_ListTensor(OpTest):
    def setUp(self):
        self.op_type = "slice"
        self.config()

        ends_tensor = []
        for index, ele in enumerate(self.ends):
            ends_tensor.append(("y" + str(index), np.ones(
                (1)).astype('int32') * ele))

        self.inputs = {
            'Input': self.input,
            "StartsTensor": np.array(
                self.starts, dtype="int32"),
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
        self.input = np.random.random([3, 4, 5, 6]).astype("float64")
        self.starts = [1, 0, 2]
        self.ends = [3, 3, 4]
        self.axes = [0, 1, 2]
        self.infer_flags = [-1, -1, -1]
        self.out = self.input[1:3, 0:3, 2:4, :]

        self.ends_infer = [-1, 3, 4]

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['Input'], 'Out', max_relative_error=0.006)


# Test CUDA float16
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFP16(OpTest):
    def setUp(self):
        self.op_type = "slice"
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
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            self.check_output_with_place(place, atol=1e-5)

    def test_check_grad_normal(self):
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            self.check_grad_with_place(
                place, ['Input'], 'Out', max_relative_error=0.006)


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFP16_2(OpTest):
    def setUp(self):
        self.op_type = "slice"
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
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            self.check_output_with_place(place, atol=1e-5)

    def test_check_grad_normal(self):
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            self.check_grad_with_place(
                place, ['Input'],
                'Out',
                max_relative_error=0.006,
                numeric_grad_delta=0.5)


# Test python API
class TestSliceAPI(unittest.TestCase):
    def test_1(self):
        input = np.random.random([3, 4, 5, 6]).astype("float64")
        minus_1 = fluid.layers.fill_constant([1], "int32", -1)
        minus_3 = fluid.layers.fill_constant([1], "int32", -3)
        starts = fluid.layers.data(
            name='starts', shape=[1, 3], append_batch_size=False)
        ends = fluid.layers.data(
            name='ends', shape=[3], append_batch_size=False)

        x = fluid.layers.data(
            name="x",
            shape=[3, 4, 5, 6],
            append_batch_size=False,
            dtype="float64")

        out_1 = fluid.layers.slice(
            x, axes=[0, 1, 2], starts=[-3, 0, 2], ends=[3, 100, -1])
        out_2 = fluid.layers.slice(
            x, axes=[0, 1, 3], starts=[minus_3, 0, 2], ends=[3, 100, -1])
        out_3 = fluid.layers.slice(
            x, axes=[0, 1, 3], starts=[minus_3, 0, 2], ends=[3, 100, minus_1])
        out_4 = fluid.layers.slice(x, axes=[0, 1, 2], starts=starts, ends=ends)

        out_5 = x[-3:3, 0:100, 2:-1]
        out_6 = x[minus_3:3, 0:100, :, 2:-1]
        out_7 = x[minus_1, 0:100, :, 2:minus_1]

        exe = fluid.Executor(place=fluid.CPUPlace())
        res_1, res_2, res_3, res_4, res_5, res_6, res_7 = exe.run(
            fluid.default_main_program(),
            feed={
                "x": input,
                'starts': np.array([-3, 0, 2]).astype("int32"),
                'ends': np.array([3, 100, -1]).astype("int32")
            },
            fetch_list=[out_1, out_2, out_3, out_4, out_5, out_6, out_7])

        assert np.array_equal(res_1, input[-3:3, 0:100, 2:-1, :])
        assert np.array_equal(res_2, input[-3:3, 0:100, :, 2:-1])
        assert np.array_equal(res_3, input[-3:3, 0:100, :, 2:-1])
        assert np.array_equal(res_4, input[-3:3, 0:100, 2:-1, :])
        assert np.array_equal(res_5, input[-3:3, 0:100, 2:-1, :])
        assert np.array_equal(res_6, input[-3:3, 0:100, :, 2:-1])
        assert np.array_equal(res_7, input[-1, 0:100, :, 2:-1])


if __name__ == '__main__':
    unittest.main()
