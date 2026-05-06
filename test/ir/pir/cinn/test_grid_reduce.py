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

import unittest

import numpy
import utils

import paddle
from paddle.static import InputSpec


class TestGridReduce(unittest.TestCase):
    def eval(
        self, dy_compute, init_inputs, input_spec=None, atol=1e-3, rtol=1e-4
    ):
        paddle.seed(2024)
        inputs = init_inputs()
        dy_out = dy_compute(*inputs)

        static_compute = utils.apply_to_static(
            dy_compute, use_cinn=True, input_spec=input_spec
        )
        st_out = static_compute(*inputs)

        for a, b in zip(
            paddle.utils.flatten(dy_out), paddle.utils.flatten(st_out)
        ):
            numpy.testing.assert_allclose(a, b, atol=atol, rtol=rtol)

    def test_all_reduce(self):
        """小 grid，块数少，侥幸能 co-resident，可通过"""

        def func(x):
            return paddle.sum(x)

        def init():
            x = paddle.randn([32, 128, 256])
            return (x,)

        self.eval(func, init)

    # ----------------------------------------------------------------
    # 以下测试覆盖"大 grid"场景，当 cooperative_groups::this_grid().sync()
    # 不能真正保证 co-residency 时，这些 case 会卡死/超时。
    # gridDim.y * gridDim.x 超过 GPU 最大同时驻留块数即触发死锁。
    # ----------------------------------------------------------------

    def test_continuous_reduce_large_batch(self):
        """
        模拟 PP-LCNetV2 BatchNorm 场景：batch=500, axis=(0,2,3)。
        CINN 生成 grid=[C_tiles, N_tiles, 1]，总块数远超硬件上限，
        cooperative_groups::this_grid().sync() 会卡死。
        """

        def func(x):
            return paddle.sum(x, axis=(0, 2, 3))

        def init():
            x = paddle.randn([500, 64, 7, 7])
            return (x,)

        self.eval(func, init)

    def test_continuous_reduce_large_batch_fp16(self):
        """
        FP16 + 大 batch，axis=(0,2,3) 时 CINN 对每个 channel 分配单 block
        （grid=[64,1,1]），不走 grid reduce 路径。精度差异来自 CINN 把 FP32
        输入逐元素 cast 成 FP16 再累加，而 native kernel 内部用 FP32 累加器，
        属于正常 FP16 vs FP32 累加语义差，容忍度需放宽到 FP16 量化级别。

        注意：若需测试真正的 FP16 grid reduce 路径，
        应使用 test_all_reduce_fp16_large（全局 reduce 强制多 block）。
        """

        def func(x):
            x = x.cast('float16')
            out = paddle.sum(x, axis=(0, 2, 3))
            return out.cast('float32')

        def init():
            x = paddle.randn([500, 64, 7, 7])
            return (x,)

        # FP16 累加 vs FP32 累加的最大绝对误差约为 1 FP16 ULP × 元素数量级
        # 实测 max_abs_diff ≈ 0.625（@value~230），用 atol=2.0 留足裕量
        self.eval(func, init, atol=2.0, rtol=1e-2)

    def test_all_reduce_fp16_large(self):
        """
        FP16 全局 all-reduce on 大 tensor，强制 CINN 走多 block + grid reduce
        路径，真正触发 cinn_grid_reduce_sum_fp16 / cooperative launch。
        """

        def func(x):
            x = x.cast('float16')
            out = paddle.sum(x)
            return out.cast('float32')

        def init():
            x = paddle.randn([500, 3, 112, 112])
            return (x,)

        self.eval(func, init, atol=2.0, rtol=1e-2)

    def test_layer_norm_large_batch(self):
        """
        LayerNorm 内含两次 grid reduce (sum 和 sum_of_squares)，
        两个 cooperative kernel 顺序入队，第二个必然卡死（第一个尚未执行完）。
        """

        def func(x):
            n = x.shape[-1]
            mean = paddle.sum(x, axis=-1, keepdim=True) / n
            var = paddle.sum((x - mean) ** 2, axis=-1, keepdim=True) / n
            return (x - mean) / paddle.sqrt(var + 1e-5)

        def init():
            # 约 6238 个 block，远超硬件同时驻留上限
            x = paddle.randn([500, 3119, 2])
            return (x,)

        self.eval(func, init)

    def test_discrete_reduce_large(self):
        """
        离散轴 reduce，axis=(0,1,2)，触发 CINN discrete reduce path。
        大 N 保证 gridDim.y 足够大引发 co-residency 问题。
        """

        def func(x):
            return paddle.sum(x, axis=(0, 1, 2))

        def init():
            x = paddle.randn([256, 28, 28, 80])
            return (x,)

        self.eval(func, init)

    def test_multiple_reduce_same_kernel(self):
        """
        同一 fused kernel 内有多个 grid reduce（模拟 BN 训练 forward）。
        多次 cooperative_groups::this_grid().sync() 考验 sense-reversing 语义。
        """

        def func(x):
            n = 500 * 7 * 7
            sum_x = paddle.sum(x, axis=(0, 2, 3))
            sum_x2 = paddle.sum(x * x, axis=(0, 2, 3))
            mean_x = sum_x / n
            mean_x2 = sum_x2 / n
            mean_x_2 = mean_x * mean_x
            return mean_x2 - mean_x_2

        def init():
            x = paddle.randn([500, 64, 7, 7])
            return (x,)

        self.eval(func, init)

    def test_continuous_reduce_dynamic_large(self):
        """
        动态 shape + 大 batch，确保 tile config 按运行时 shape 计算，
        静态小 grid 的 tile 在运行时 block 数量膨胀触发死锁。
        """

        def func(x):
            return paddle.sum(x, axis=(0, 2, 3))

        def init():
            x = paddle.randn([500, 3, 20, 20])
            return (x,)

        input_spec = [InputSpec([-1, 3, -1, -1])]
        self.eval(func, init, input_spec)

    def test_all_reduce_very_large(self):
        """
        AllReduce 在极大 tensor 上，单次 cooperative 就超硬件容量。
        """

        def func(x):
            return paddle.sum(x)

        def init():
            # 约 8M 元素，block 数会很多
            x = paddle.randn([500, 3, 112, 112])
            return (x,)

        self.eval(func, init)


if __name__ == "__main__":
    unittest.main()
