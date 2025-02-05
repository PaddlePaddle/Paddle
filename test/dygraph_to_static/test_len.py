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
from dygraph_to_static_utils import (
    Dy2StTestBase,
    static_guard,
    test_ast_only,
    test_legacy_and_pt,
    test_pir_only,
)

import paddle
from paddle import base
from paddle.jit.dy2static import Call
from paddle.nn import clip

SEED = 2020
np.random.seed(SEED)


def len_with_tensor(x):
    x = paddle.to_tensor(x)
    x_len = len(x)
    return x_len


def len_with_dense_tensor_array(x):
    x = paddle.to_tensor(x)

    i = paddle.tensor.fill_constant(shape=[1], dtype='int64', value=0)
    arr = paddle.tensor.array_write(x, i=i)
    arr_len = len(arr)

    return arr_len


class TestLen(Dy2StTestBase):
    def setUp(self):
        self.x_data = np.random.random([10, 16]).astype('float32')
        self.init_func()

    def init_func(self):
        self.func = len_with_tensor

    def _run(self, to_static):
        if to_static:
            out = paddle.jit.to_static(self.func)(self.x_data)
        else:
            out = self.func(self.x_data)

        if isinstance(out, paddle.Tensor):
            out = out.numpy()
        return out

    @test_ast_only
    def test_len(self):
        dygraph_res = self._run(to_static=False)
        static_res = self._run(to_static=True)
        np.testing.assert_allclose(dygraph_res, static_res, rtol=1e-05)


class TestLenWithTensorArray(TestLen):
    def init_func(self):
        self.func = len_with_dense_tensor_array


# Note: Variable(SelectedRows) is not exposed directly in dygraph.
# The unittest is used to test coverage by fake transformed code.
def len_with_selected_rows(place):
    # create selected_rows variable
    non_used_initializer = paddle.nn.initializer.Constant(0.0)
    var = paddle.static.create_parameter(
        name="X",
        dtype="float32",
        shape=[5, 20],
    )
    selected_var = (
        paddle.base.libpaddle.pir.create_selected_rows_type_by_dense_tensor(
            var.type()
        )
    )
    var.set_type(selected_var)
    # y is Variable(SelectedRows)
    y = clip.merge_selected_rows(var)
    y_len = Call(len)(y)

    # z is inner tensor with shape [4, 2]
    z = clip.get_tensor_from_selected_rows(y)
    z_len = paddle.shape(z)[0]

    # set data for selected_rows
    x_rows = [0, 2, 2, 4, 19]
    row_numel = 2
    np_array = np.ones((len(x_rows), row_numel)).astype("float32")

    x_var = paddle.static.global_scope().var("X").get_selected_rows()
    x_var.set_rows(x_rows)
    x_var.set_height(20)
    x_tensor = x_var.get_tensor()
    x_tensor.set(np_array, place)

    exe = paddle.static.Executor(place=place)
    result = exe.run(
        paddle.static.default_main_program(), fetch_list=[y_len, z_len]
    )
    return result


def legacy_len_with_selected_rows(place):
    block = paddle.static.default_main_program().global_block()
    # create selected_rows variable
    var = block.create_var(
        name="X",
        dtype="float32",
        shape=[-1],
        persistable=True,
        type=base.core.VarDesc.VarType.SELECTED_ROWS,
    )
    # y is Variable(SelectedRows)
    y = clip.merge_selected_rows(var)
    y_len = Call(len)(y)

    # z is inner tensor with shape [4, 2]
    z = clip.get_tensor_from_selected_rows(y)
    z_len = Call(len)(z)

    # set data for selected_rows
    x_rows = [0, 2, 2, 4, 19]
    row_numel = 2
    np_array = np.ones((len(x_rows), row_numel)).astype("float32")

    x_var = paddle.static.global_scope().var("X").get_selected_rows()
    x_var.set_rows(x_rows)
    x_var.set_height(20)
    x_tensor = x_var.get_tensor()
    x_tensor.set(np_array, place)

    exe = paddle.static.Executor(place=place)
    result = exe.run(
        paddle.static.default_main_program(), fetch_list=[y_len, z_len]
    )
    return result


class TestLenWithSelectedRows(Dy2StTestBase):
    def setUp(self):
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    @test_ast_only
    @test_legacy_and_pt
    def test_len_legacy(self):
        with static_guard():
            (
                selected_rows_var_len,
                var_tensor_len,
            ) = legacy_len_with_selected_rows(self.place)
        self.assertEqual(selected_rows_var_len, var_tensor_len)

    @test_ast_only
    @test_pir_only
    def test_len(self):
        with static_guard():
            selected_rows_var_len, var_tensor_len = len_with_selected_rows(
                self.place
            )
        self.assertEqual(selected_rows_var_len, var_tensor_len)


if __name__ == '__main__':
    unittest.main()
