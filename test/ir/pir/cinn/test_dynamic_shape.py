# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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


import os
import unittest

import numpy

os.environ['FLAGS_prim_all'] = 'true'
os.environ['FLAGS_prim_enable_dynamic'] = 'true'
os.environ['FLAGS_use_cinn'] = '1'
os.environ['FLAGS_deny_cinn_ops'] = 'slice;'

import paddle


def generate_input_spec(rank_dtype_list):
    input_spec = []
    for rank, dtype in rank_dtype_list:
        input_spec.append(
            paddle.static.InputSpec(shape=[None] * rank, dtype=dtype)
        )
    return input_spec


class TestTrivialFusion(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def compare_result(self, dy_compute, input_spec, data_init):
        inputs = data_init()
        dy_out = dy_compute(*inputs)
        static_compute = paddle.jit.to_static(
            full_graph=True,
            backend="CINN",
            input_spec=input_spec,
        )(dy_compute)
        st_out = static_compute(*inputs)
        if isinstance(dy_out, paddle.Tensor):
            numpy.testing.assert_allclose(dy_out, st_out, atol=1e-5, rtol=1e-6)
            return
        for d, s in zip(dy_out, st_out):
            numpy.testing.assert_allclose(d, s, atol=1e-5, rtol=1e-6)

    def test_simple_trivial_fusions(self):
        def func(x):
            x = x * 2
            x = x + 1
            x = paddle.nn.functional.relu(x)
            x = paddle.transpose(x, perm=[0, 2, 1])
            x = x.reshape((-1, 128))
            return x

        def init():
            x = paddle.rand((32, 32, 128))
            return (x,)

        input_spec = generate_input_spec([(3, 'float32')])
        self.compare_result(func, input_spec, init)

    def test_trivial_fusion_slice_and_concat(self):
        def func(x, y):
            x = x * 2
            y = y * 2
            x = x[:, :, :64]
            y = y[:, :, :64]
            z = paddle.concat([x, y], axis=-1)
            return z

        def init():
            x = paddle.rand((32, 32, 128))
            y = paddle.rand((32, 32, 128))
            return (x, y)

        input_spec = generate_input_spec([(3, 'float32'), (3, 'float32')])
        self.compare_result(func, input_spec, init)

    def test_trivial_fusion_gather_nd(self):
        def func(x, y):
            x = x * 2
            output = paddle.gather_nd(x, y)
            return output

        def init():
            x = paddle.to_tensor(
                [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]
            )
            index = paddle.to_tensor([[0, 1]])
            return (x, index)

        input_spec = [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 2], dtype='int32'),
        ]
        self.compare_result(func, input_spec, init)

    def test_broadcast(self):
        def func(x, y):
            output = x + y
            return output

        def init():
            x = paddle.rand((32, 1))
            y = paddle.rand((1, 32))
            return (x, y)

        input_spec = generate_input_spec([(2, 'float32'), (2, 'float32')])
        self.compare_result(func, input_spec, init)

    def test_broadcast_tree(self):
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
            var_769 = paddle.full(
                [20, 32, var_4.shape[0], 8, 24], 0.1, "float32"
            )
            var_kwarg_middle_31 = paddle.rand(
                [20, var_9.shape[1], 8, 24], "float32"
            )
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

        self.compare_result(func, input_spec(), init)


if __name__ == "__main__":
    unittest.main()
