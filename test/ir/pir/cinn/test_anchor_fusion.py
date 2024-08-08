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


class TestAnchorFusion(unittest.TestCase):
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
            numpy.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-6)

    def test_anchor_fusion_1(self):
        def func(x):
            x = x * 3
            a = x + 1
            b = x + 2
            return a, b

        def init():
            x = paddle.rand((32, 32, 128))
            return (x,)

        self.compare_result(func, None, init)

    def test_anchor_fusion_2(self):
        def func(x):
            x = x * 2
            a = x + 1
            c = paddle.sum(a, axis=-1)
            b = x + 2
            return b, c

        def init():
            x = paddle.ones((32, 32, 128))
            return (x,)

        self.compare_result(func, None, init)

    def test_anchor_fusion_3(self):
        def func(x):
            x = x * 2
            a = x + 1
            c = paddle.sum(a, axis=-1)
            b = x + 2
            d = paddle.sum(b, axis=-1)
            return c, d

        def init():
            x = paddle.rand((32, 32, 128))
            return (x,)

        self.compare_result(func, None, init)

    def test_anchor_fusion_4(self):
        def func(x):
            m = x + 1
            a = m + 1
            b = m + 2
            c = a * 2
            d = a / 2
            return b, c, d

        def init():
            x = paddle.rand((32, 32, 128))
            return (x,)

        self.compare_result(func, None, init)

    def test_anchor_fusion_5(self):
        #     R
        #    / \
        #   T   T
        def func(x):
            a = paddle.sum(x, axis=-1)
            b = a - 3
            c = a * 2
            return b, c

        def init():
            x = paddle.rand((32, 32, 128))
            return (x,)

        self.compare_result(func, None, init)

    def test_anchor_fusion_6(self):
        #      T
        #     / \
        #    B   R
        #   /     \
        #  T       T
        def func(x):
            a = x + 1
            b = paddle.expand(a, [10, 32, 32, 128]) / 2
            c = paddle.sum(a, axis=-1)
            d = c * 3
            return b, d

        def init():
            x = paddle.rand((32, 32, 128))
            return (x,)

        self.compare_result(func, None, init)


if __name__ == "__main__":
    unittest.main()
