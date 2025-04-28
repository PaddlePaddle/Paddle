# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import numpy

import paddle


def init():
    var_1 = paddle.rand([32], dtype="float32")
    var_2 = paddle.rand([32], dtype="float32")
    var_3 = paddle.rand([32], dtype="float32")
    return (var_1, var_2, var_3)


def input_spec():
    return [
        paddle.static.InputSpec(shape=[None], dtype='float32'),  # S0
        paddle.static.InputSpec(shape=[None], dtype='float32'),  # S1
        paddle.static.InputSpec(shape=[None], dtype='float32'),  # S2
    ]


def func(var_1, var_2, var_3):
    var_4 = paddle.reshape(var_1, [-1, 32])  # Div(S0, 32)
    var_5 = paddle.reshape(var_2, [-1, 32])  # Div(S1, 32)
    var_6 = paddle.reshape(var_3, [-1, 32])  # Div(S2, 32)
    # Broadcast(Div(S0, 32), Div(S1, 32), Div(S2, 32)
    var_7 = var_4 + var_5 + var_6
    # Mul(Broadcast(Div(S0, 32), Div(S1, 32), Div(S2, 32)), 32)
    var_9 = var_7.reshape([1, -1, 1, 1])

    var_752 = paddle.full([20, var_9.shape[1], 8, 24], 0.1, "float32")
    var_kwarg_var_10744 = var_2
    var_769 = paddle.full([20, 32, var_4.shape[0], 8, 24], 0.1, "float32")
    var_kwarg_middle_31 = paddle.rand([20, var_9.shape[1], 8, 24], "float32")
    var_kwarg_middle_31[:] = 0.1
    var_kwarg_middle_30 = paddle.full([20, 32, 1, 1, 1], 0.1, "float32")

    var_812 = paddle.full(shape=[], dtype='float32', fill_value=0.0)
    var_814 = paddle.expand(var_812, var_kwarg_middle_31.shape)
    var_815 = paddle.greater_than(var_kwarg_middle_31, var_814)
    var_816 = paddle.cast(var_815, dtype='float32')
    var_817 = var_816 * var_752
    var_818 = paddle.reshape(var_817, [20, 32, -1, 8, 24])
    var_819 = paddle.reshape(var_kwarg_var_10744, [32, -1, 1, 1])
    var_820 = paddle.full(shape=[20, 32, 1, 1, 1], fill_value=1e-05)
    var_821 = var_kwarg_middle_30 + var_820
    var_822 = paddle.full(shape=[20, 32, 1, 1, 1], fill_value=1.0)
    var_823 = var_822 / var_821
    var_824 = paddle.sqrt(var_823)
    var_827 = var_818 * var_819
    var_830 = var_824 * var_827
    var_831 = paddle.sum(var_830, keepdim=True, axis=[2, 3, 4])
    var_834 = var_831 * var_769
    var_837 = var_834 * var_827
    var_838 = paddle.sum(var_837, keepdim=True, axis=[2, 3, 4])
    return var_818, var_824, var_838, var_831, var_830


class TestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def compare_result(self, dy_compute, data_init):
        static_compute = paddle.jit.to_static(
            full_graph=True, input_spec=input_spec(), backend="CINN"
        )(dy_compute)
        inputs = data_init()
        dy_out = dy_compute(*inputs)
        st_out = static_compute(*inputs)
        if isinstance(dy_out, paddle.Tensor):
            numpy.testing.assert_allclose(dy_out, st_out, atol=1e-5, rtol=1e-6)
            return
        for d, s in zip(dy_out, st_out):
            numpy.testing.assert_allclose(d, s, atol=1e-5, rtol=1e-6)

    def test_case(self):
        self.compare_result(func, init)


if __name__ == "__main__":
    unittest.main()
