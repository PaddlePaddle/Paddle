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

os.environ['FLAGS_cinn_new_group_scheduler'] = '1'
os.environ['FLAGS_group_schedule_tiling_first'] = '1'
os.environ['FLAGS_prim_all'] = 'true'
os.environ['FLAGS_prim_enable_dynamic'] = 'true'
os.environ['FLAGS_print_ir'] = '1'
os.environ['FLAGS_enable_pir_api'] = '1'
os.environ['FLAGS_use_cinn'] = '1'
os.environ['FLAGS_cinn_bucket_compile'] = '1'
os.environ['FLAGS_cinn_new_cluster_op_method'] = '1'
os.environ['FLAGS_deny_cinn_ops'] = 'slice;'

import paddle

build_strategy = paddle.static.BuildStrategy()
build_strategy.build_cinn_pass = True


def generate_input_spec(rank_dtype_list):
    input_spec = []
    for rank, dtype in rank_dtype_list:
        input_spec.append(
            paddle.static.InputSpec(shape=[None] * rank, dtype=dtype)
        )
    return input_spec


class TestReduceFusion(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def compare_result(self, dy_compute, input_spec, data_init):
        inputs = data_init()
        dy_out = dy_compute(*inputs)
        static_compute = paddle.jit.to_static(
            full_graph=True,
            build_strategy=build_strategy,
            input_spec=input_spec,
        )(dy_compute)
        st_out = static_compute(*inputs)
        for a, b in zip(
            paddle.utils.flatten(dy_out), paddle.utils.flatten(st_out)
        ):
            numpy.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-5)

    def test_reduce_tree_grown(self):
        #  R -> B -> R
        def func(x):
            b = paddle.max(x, axis=-1)
            c = paddle.expand(b, [128, 32, 32])
            d = paddle.sum(c, axis=0)
            return d

        def init():
            x = paddle.rand((32, 32, 128))
            return (x,)

        self.compare_result(func, None, init)

    def test_reduce_broadcast_fusion(self):
        #  R -> B
        def func(x):
            b = paddle.max(x, axis=-1)
            c = paddle.expand(b, [128, 32, 32])
            return c

        def init():
            x = paddle.rand((32, 32, 128))
            return (x,)

        self.compare_result(func, None, init)

    def test_reduce_tree_plus_trivial(self):
        #  T -> R -> T
        def func(x):
            a = x + 1
            b = paddle.max(a, axis=-1)
            c = b / 3.0
            return c

        def init():
            x = paddle.rand((32, 32, 128))
            return (x,)

        self.compare_result(func, None, init)

    def test_reduce_fusion_without_axis_reuse(self):
        #     R
        #    / \
        #   T   T
        #    \ /
        #     T
        #     |
        #     B
        #     |
        #     R
        def func(x):
            b = paddle.max(x, axis=-1)
            c = b * 2
            d = b / 2
            e = c + d
            f = paddle.expand(e, [96, 32, 32])
            g = paddle.sum(f, axis=0)
            return g

        def init():
            x = paddle.rand((32, 32, 128))
            return (x,)

        self.compare_result(func, None, init)

    def test_reduce_all_reshape(self):
        # R(reduce all) -> reshape
        def func(x):
            a = paddle.max(x, axis=[0, 1, 2, 3], keepdim=False)
            b = paddle.reshape(a, [1])
            return b

        def init():
            x = paddle.rand((1, 1, 128, 128))
            return (x,)

        self.compare_result(func, None, init)

    def test_cast_int32_reduce(self):
        def func(x):
            a = paddle.cast(x, dtype='int32')
            b = paddle.max(a, axis=[2], keepdim=False)
            return b

        def init():
            x = paddle.rand((3, 128, 96), dtype='float32')
            return (x,)

        self.compare_result(func, None, init)


if __name__ == "__main__":
    unittest.main()
