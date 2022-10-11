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

import unittest

import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import declarative
from paddle.fluid.dygraph.dygraph_to_static import convert_call

SEED = 2020
np.random.seed(SEED)
paddle.enable_static()


def len_with_tensor(x):
    x = fluid.dygraph.to_variable(x)
    x_len = len(x)
    return x_len


def len_with_lod_tensor_array(x):
    x = fluid.dygraph.to_variable(x)

    i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)
    arr = fluid.layers.array_write(x, i=i)
    arr_len = len(arr)

    return arr_len


class TestLen(unittest.TestCase):

    def setUp(self):
        self.place = fluid.CUDAPlace(
            0) if fluid.is_compiled_with_cuda() else fluid.CPUPlace()
        self.x_data = np.random.random([10, 16]).astype('float32')
        self.init_func()

    def init_func(self):
        self.func = len_with_tensor

    def _run(self, to_static):
        with fluid.dygraph.guard(self.place):
            if to_static:
                out = declarative(self.func)(self.x_data)
            else:
                out = self.func(self.x_data)

            if isinstance(out, fluid.core.VarBase):
                out = out.numpy()
            return out

    def test_len(self):
        dygraph_res = self._run(to_static=False)
        static_res = self._run(to_static=True)
        np.testing.assert_allclose(dygraph_res, static_res, rtol=1e-05)


class TestLenWithTensorArray(TestLen):

    def init_func(self):
        self.func = len_with_lod_tensor_array


# Note: Variable(SelectedRows) is not exposed directly in dygraph.
# The unittest is used to test coverage by fake transformed code.
def len_with_selected_rows(place):
    block = fluid.default_main_program().global_block()
    # create selected_rows variable
    var = block.create_var(name="X",
                           dtype="float32",
                           persistable=True,
                           type=fluid.core.VarDesc.VarType.SELECTED_ROWS)
    # y is Variable(SelectedRows)
    y = fluid.layers.merge_selected_rows(var)
    y_len = convert_call(len)(y)

    # z is inner tensor with shape [4, 2]
    z = fluid.layers.get_tensor_from_selected_rows(y)
    z_len = convert_call(len)(z)

    # set data for selected_rows
    x_rows = [0, 2, 2, 4, 19]
    row_numel = 2
    np_array = np.ones((len(x_rows), row_numel)).astype("float32")

    x_var = fluid.global_scope().var("X").get_selected_rows()
    x_var.set_rows(x_rows)
    x_var.set_height(20)
    x_tensor = x_var.get_tensor()
    x_tensor.set(np_array, place)

    exe = fluid.Executor(place=place)
    result = exe.run(fluid.default_main_program(), fetch_list=[y_len, z_len])
    return result


class TestLenWithSelectedRows(unittest.TestCase):

    def setUp(self):
        self.place = fluid.CUDAPlace(
            0) if fluid.is_compiled_with_cuda() else fluid.CPUPlace()

    def test_len(self):
        selected_rows_var_len, var_tensor_len = len_with_selected_rows(
            self.place)
        self.assertEqual(selected_rows_var_len, var_tensor_len)


if __name__ == '__main__':
    unittest.main()
