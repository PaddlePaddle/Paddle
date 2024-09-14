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
import utils

os.environ['FLAGS_cinn_new_group_scheduler'] = '1'
os.environ['FLAGS_group_schedule_tiling_first'] = '1'
os.environ['FLAGS_prim_all'] = 'true'
os.environ['FLAGS_prim_enable_dynamic'] = 'true'
os.environ['FLAGS_print_ir'] = '1'
os.environ['FLAGS_enable_pir_api'] = '1'
os.environ['FLAGS_use_cinn'] = '1'
os.environ['FLAGS_cinn_bucket_compile'] = '1'
os.environ['FLAGS_cinn_new_cluster_op_method'] = '1'

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

    def check_accuracy_and_kernel_num(
        self, data_init, dy_compute, kernel_num=None, input_spec=None
    ):
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
            numpy.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-6)
        if kernel_num is not None:
            utils.check_jit_kernel_number(static_compute, kernel_num)

    # def test_identiy_iters_fusion(self):
    #     #        T
    #     #      / | \
    #     #     /  |  \
    #     #    T   T   T
    #     #   / \     / \
    #     #  T   T   T   T
    #     def func(x):
    #         x = x * 3
    #         a = x + 1
    #         d = paddle.sqrt(a)
    #         e = paddle.ceil(a)
    #         b = x - 2
    #         c = paddle.exp(x)
    #         f = paddle.log(c)
    #         g = paddle.sign(c)
    #         return d, e, b, f, g

    #     def init():
    #         x = paddle.rand((32, 32, 128))
    #         return (x,)

    #     self.check_accuracy_and_kernel_num(init, func, kernel_num=1)

    # def test_transpose_iters_fusion(self):
    #     #     Tranpose
    #     #      /  \
    #     #     T   Tranpose
    #     #    / \
    #     #   T  Tranpose
    #     def func(x):
    #         x = paddle.transpose(x, [2, 0, 1, 3])
    #         a = x + 1
    #         b = paddle.transpose(x, [3, 1, 0, 2])
    #         c = a / 3
    #         d = paddle.transpose(a, [0, 2, 3, 1])
    #         return b, c, d

    #     def init():
    #         x = paddle.ones((16, 32, 64, 128))
    #         return (x,)

    #     self.check_accuracy_and_kernel_num(init, func, kernel_num=1)

    # def test_append_iters_fusion(self):
    #     #       T
    #     #     /   \
    #     #    S     B
    #     #   / \   / \
    #     #  S   B S   T
    #     def func(x):
    #         x = x * 2
    #         a = x[0, :]  # shape=[64, 128]
    #         b = paddle.expand(x, [16, 32, 64, 128])
    #         c = a[:, 0]  # shape=[64]
    #         d = paddle.expand(a, [8, 16, 32, 64, 128])
    #         e = b[0, :, 0, :]  # shape=[32,128]
    #         f = paddle.exp(b)
    #         return c, d, e, f

    #     def init():
    #         x = paddle.rand((32, 64, 128))
    #         return (x,)

    #     self.check_accuracy_and_kernel_num(init, func, kernel_num=1)

    def test_trivial_reuse_iters_fusion(self):
        #     T
        #    / \
        #   S   B
        #   |   |
        #   B   S
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

    # def test_anchor_fusion_5(self):
    #     #     R
    #     #    / \
    #     #   T   T
    #     def func(x):
    #         a = paddle.sum(x, axis=-1)
    #         b = a - 3
    #         c = a * 2
    #         return b, c

    #     def init():
    #         x = paddle.rand((32, 32, 128))
    #         return (x,)

    #     self.compare_result(func, None, init)

    # def test_anchor_fusion_6(self):
    #     #      T
    #     #     / \
    #     #    B   R
    #     #   /     \
    #     #  T       T
    #     def func(x):
    #         a = x + 1
    #         b = paddle.expand(a, [10, 32, 32, 128]) / 2
    #         c = paddle.sum(a, axis=-1)
    #         d = c * 3
    #         return b, d

    #     def init():
    #         x = paddle.rand((32, 32, 128))
    #         return (x,)

    #     self.compare_result(func, None, init)

    # def test_anchor_fusion_7(self):
    #     #      T
    #     #     / \
    #     #    T   Transpose
    #     def func(x):
    #         a = x + 1
    #         b = a * 3
    #         c = paddle.transpose(a, [1, 0, 2])
    #         # c = a / 2
    #         return b, c

    #     def init():
    #         x = paddle.rand((32, 64, 128))
    #         return (x,)

    #     self.compare_result(func, None, init)

    # def test_fusion_iters(self):
    #     #       T
    #     #     / | \
    #     #    /  |  \
    #     #   T   S   B
    #     def func(x):
    #         a = x + 2
    #         b = a * 3
    #         c = a[0, :]
    #         d = paddle.expand(a, [16, 32, 64])
    #         return b, c, d

    #     def init():
    #         x = paddle.rand((32, 64))
    #         return (x,)

    #     self.compare_result(func, None, init)

    # def test_fusion_iters(self):
    #     #      T
    #     #     / \
    #     #    T   R
    #     #       / \
    #     #      T   B
    #     #           \
    #     #            R
    #     def func(x):
    #         a = x + 2
    #         b = a * 3
    #         c = paddle.max(a, axis=0)
    #         d = c - 4
    #         e = paddle.expand(c, [1024, 32, 64])
    #         f = paddle.sum(e, axis=0)
    #         return b, d, f

    #     def init():
    #         x = paddle.rand((1024, 32, 64))
    #         return (x,)

    #     self.compare_result(func, None, init)

    # def test_fusion_iters(self):
    #     #     T   T
    #     #      \ /
    #     #       T
    #     #    / | | \
    #     #   /  | |  \
    #     #  T   S R  Transpose
    #     #      | |
    #     #      B B
    #     #        |
    #     #        R
    #     def func(x, y):
    #         a = x + 1
    #         b = y * 2
    #         c = a + b
    #         d = paddle.transpose(c, [1, 0, 2])
    #         e = c[0, :]
    #         f = paddle.expand(e, [16, 32, 64])
    #         g = paddle.max(c, axis=0)
    #         h = paddle.expand(g, [16, 32, 64])
    #         i = paddle.sum(h, axis=0)
    #         return c, d, f, i

    #     def init():
    #         x = paddle.rand((16, 32, 64))
    #         y = paddle.rand((16, 32, 64))
    #         return (x, y)

    #     self.compare_result(func, None, init)

    # def test_batch_norm(self):
    #     self.batch_norm = paddle.nn.layer.norm.BatchNorm(2048)

    #     def func(x):
    #         a = self.batch_norm(x)
    #         return a

    #     def init():
    #         x = paddle.rand((128, 2048, 7, 7))
    #         return (x,)

    #     self.compare_result(func, None, init)


if __name__ == "__main__":
    unittest.main()
