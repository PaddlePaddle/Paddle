# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

import paddle
import paddle.distributed as dist
import paddle.nn.functional as F


class TestElementwiseApiForSemiAutoParallel:
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

        paddle.seed(self._seed)
        np.random.seed(self._seed)

    def check_tensor_eq(self, a, b, rtol=1e-5):
        np1 = a.numpy()
        np2 = b.numpy()
        np.testing.assert_allclose(np1, np2, rtol=rtol, verbose=True)

    def test_unary_body(
        self, x_shape, out_shape, x_specs, unary_func, rtol=1e-5
    ):
        x = paddle.randn(x_shape, self._dtype)
        x.stop_gradient = False

        x_dist_attr = dist.DistAttr(mesh=self._mesh, sharding_specs=x_specs)

        dist_x = dist.shard_tensor(x, dist_attr=x_dist_attr)
        dist_x.stop_gradient = False

        dist_out = unary_func(dist_x)
        out = unary_func(x)
        self.check_tensor_eq(out, dist_out, rtol=rtol)

        dist_out.backward()
        out.backward()
        self.check_tensor_eq(x.grad, dist_x.grad, rtol=rtol)

    def test_binary_body(
        self,
        x_shape,
        y_shape,
        out_shape,
        x_specs,
        y_specs,
        binary_func,
        rtol=1e-5,
    ):
        x = paddle.randn(x_shape, self._dtype)
        y = paddle.randn(y_shape, self._dtype)
        x.stop_gradient = False
        y.stop_gradient = False

        x_dist_attr = dist.DistAttr(mesh=self._mesh, sharding_specs=x_specs)
        y_dist_attr = dist.DistAttr(mesh=self._mesh, sharding_specs=y_specs)

        dist_x = dist.shard_tensor(x, dist_attr=x_dist_attr)
        dist_y = dist.shard_tensor(y, dist_attr=y_dist_attr)
        dist_x.stop_gradient = False
        dist_y.stop_gradient = False

        dist_out = binary_func(dist_x, dist_y)
        out = binary_func(x, y)
        self.check_tensor_eq(out, dist_out, rtol=rtol)

        dist_out.backward()
        out.backward()
        self.check_tensor_eq(x.grad, dist_x.grad, rtol=rtol)
        self.check_tensor_eq(y.grad, dist_y.grad, rtol=rtol)

    def test_add_x_shard(self):
        self.test_binary_body(
            x_shape=[16, 32],
            y_shape=[16, 32],
            out_shape=[16, 32],
            x_specs=['x', None],
            y_specs=[None, None],
            binary_func=paddle.add,
        )

    def test_sub_x_shard(self):
        self.test_binary_body(
            x_shape=[16, 32],
            y_shape=[16, 32],
            out_shape=[16, 32],
            x_specs=['x', None],
            y_specs=[None, None],
            binary_func=paddle.subtract,
        )

    def test_add_x_shard_broadcast(self):
        self.test_binary_body(
            x_shape=[16, 32],
            y_shape=[2, 16, 32],
            out_shape=[2, 16, 32],
            x_specs=['x', None],
            y_specs=[None, None, None],
            binary_func=paddle.add,
        )

    def test_add_x_y_shard(self):
        if self._backend == "cpu":
            return

        self.test_binary_body(
            x_shape=[16, 32],
            y_shape=[16, 32],
            out_shape=[16, 32],
            x_specs=['x', None],
            y_specs=[None, 'x'],
            binary_func=paddle.add,
        )

    def test_add_x_y_shard_broadcast(self):
        if self._backend == "cpu":
            return

        self.test_binary_body(
            x_shape=[4, 16, 32],
            y_shape=[16, 32],
            out_shape=[4, 16, 32],
            x_specs=['x', None, None],
            y_specs=[None, None],
            binary_func=paddle.add,
        )

    def test_sub_x_y_shard_broadcast(self):
        if self._backend == "cpu":
            return

        self.test_binary_body(
            x_shape=[4, 16, 32],
            y_shape=[16, 32],
            out_shape=[4, 16, 32],
            x_specs=['x', None, None],
            y_specs=[None, None],
            binary_func=paddle.subtract,
        )

    def test_square_x_shard(self):
        self.test_unary_body(
            x_shape=[4, 16],
            out_shape=[4, 16],
            x_specs=['x', None],
            unary_func=paddle.square,
        )

    def test_relu_x_shard(self):
        self.test_unary_body(
            x_shape=[4, 16],
            out_shape=[4, 16],
            x_specs=['x', None],
            unary_func=F.relu,
        )

    def test_maximum_x_shard(self):
        self.test_binary_body(
            x_shape=[16, 32],
            y_shape=[16, 32],
            out_shape=[16, 32],
            x_specs=['x', None],
            y_specs=[None, None],
            binary_func=paddle.maximum,
        )

    def test_maximum_x_shard_broadcast(self):
        self.test_binary_body(
            x_shape=[16, 32],
            y_shape=[2, 16, 32],
            out_shape=[2, 16, 32],
            x_specs=['x', None],
            y_specs=[None, None, None],
            binary_func=paddle.maximum,
        )

    def test_maximum_x_y_shard(self):
        if self._backend == "cpu":
            return

        self.test_binary_body(
            x_shape=[16, 32],
            y_shape=[16, 32],
            out_shape=[16, 32],
            x_specs=['x', None],
            y_specs=[None, 'x'],
            binary_func=paddle.maximum,
        )

    def test_maximum_x_y_shard_broadcast(self):
        self.test_binary_body(
            x_shape=[4, 16, 32],
            y_shape=[16, 32],
            out_shape=[4, 16, 32],
            x_specs=['x', None, None],
            y_specs=[None, None],
            binary_func=paddle.maximum,
        )

    def test_multiply_x_shard(self):
        self.test_binary_body(
            x_shape=[16, 32],
            y_shape=[16, 32],
            out_shape=[16, 32],
            x_specs=['x', None],
            y_specs=[None, None],
            binary_func=paddle.multiply,
        )

    def test_multiply_x_shard_broadcast(self):
        self.test_binary_body(
            x_shape=[16, 32],
            y_shape=[2, 16, 32],
            out_shape=[2, 16, 32],
            x_specs=['x', None],
            y_specs=[None, None, None],
            binary_func=paddle.multiply,
        )

    def test_multiply_x_y_shard(self):
        if self._backend == "cpu":
            return
        self.test_binary_body(
            x_shape=[16, 32],
            y_shape=[16, 32],
            out_shape=[16, 32],
            x_specs=['x', None],
            y_specs=[None, 'x'],
            binary_func=paddle.multiply,
        )

    def test_multiply_x_y_shard_broadcast(self):
        self.test_binary_body(
            x_shape=[4, 16, 32],
            y_shape=[16, 32],
            out_shape=[4, 16, 32],
            x_specs=['x', None, None],
            y_specs=[None, None],
            binary_func=paddle.multiply,
        )

    def test_divide_x_shard(self):
        self.test_binary_body(
            x_shape=[16, 32],
            y_shape=[16, 32],
            out_shape=[16, 32],
            x_specs=['x', None],
            y_specs=[None, None],
            binary_func=paddle.divide,
        )

    def test_divide_x_shard_broadcast(self):
        self.test_binary_body(
            x_shape=[16, 32],
            y_shape=[2, 16, 32],
            out_shape=[2, 16, 32],
            x_specs=['x', None],
            y_specs=[None, None, None],
            binary_func=paddle.divide,
        )

    def test_divide_x_y_shard(self):
        if self._backend == "cpu":
            return
        self.test_binary_body(
            x_shape=[16, 32],
            y_shape=[16, 32],
            out_shape=[16, 32],
            x_specs=['x', None],
            y_specs=[None, 'x'],
            binary_func=paddle.divide,
        )

    def test_divide_x_y_shard_broadcast(self):
        self.test_binary_body(
            x_shape=[4, 16, 32],
            y_shape=[16, 32],
            out_shape=[4, 16, 32],
            x_specs=['x', None, None],
            y_specs=[None, None],
            binary_func=paddle.divide,
        )

    def test_bitwise_and_x_shard(self):
        self.test_binary_body(
            x_shape=[16, 32],
            y_shape=[16, 32],
            out_shape=[16, 32],
            x_specs=['x', None],
            y_specs=[None, None],
            binary_func=paddle.bitwise_and,
        )

    def test_bitwise_and_x_shard_broadcast(self):
        self.test_binary_body(
            x_shape=[16, 32],
            y_shape=[2, 16, 32],
            out_shape=[2, 16, 32],
            x_specs=['x', None],
            y_specs=[None, None, None],
            binary_func=paddle.bitwise_and,
        )

    def test_bitwise_and_x_y_shard(self):
        if self._backend == "cpu":
            return
        self.test_binary_body(
            x_shape=[16, 32],
            y_shape=[16, 32],
            out_shape=[16, 32],
            x_specs=['x', None],
            y_specs=[None, 'x'],
            binary_func=paddle.bitwise_and,
        )

    def test_bitwise_and_x_y_shard_broadcast(self):
        self.test_binary_body(
            x_shape=[4, 16, 32],
            y_shape=[16, 32],
            out_shape=[4, 16, 32],
            x_specs=['x', None, None],
            y_specs=[None, None],
            binary_func=paddle.bitwise_and,
        )

    def test_elementwise_pow_x_shard(self):
        self.test_binary_body(
            x_shape=[16, 32],
            y_shape=[16, 32],
            out_shape=[16, 32],
            x_specs=['x', None],
            y_specs=[None, None],
            binary_func=paddle.pow,
        )

    def test_elementwise_pow_x_shard_broadcast(self):
        self.test_binary_body(
            x_shape=[16, 32],
            y_shape=[2, 16, 32],
            out_shape=[2, 16, 32],
            x_specs=['x', None],
            y_specs=[None, None, None],
            binary_func=paddle.pow,
        )

    def test_elementwise_pow_x_y_shard(self):
        if self._backend == "cpu":
            return
        self.test_binary_body(
            x_shape=[16, 32],
            y_shape=[16, 32],
            out_shape=[16, 32],
            x_specs=['x', None],
            y_specs=[None, 'x'],
            binary_func=paddle.pow,
        )

    def test_elementwise_pow_x_y_shard_broadcast(self):
        self.test_binary_body(
            x_shape=[4, 16, 32],
            y_shape=[16, 32],
            out_shape=[4, 16, 32],
            x_specs=['x', None, None],
            y_specs=[None, None],
            binary_func=paddle.pow,
        )

    def test_equal_x_shard(self):
        self.test_binary_body(
            x_shape=[16, 32],
            y_shape=[16, 32],
            out_shape=[16, 32],
            x_specs=['x', None],
            y_specs=[None, None],
            binary_func=paddle.equal,
        )

    def test_equal_x_shard_broadcast(self):
        self.test_binary_body(
            x_shape=[16, 32],
            y_shape=[2, 16, 32],
            out_shape=[2, 16, 32],
            x_specs=['x', None],
            y_specs=[None, None, None],
            binary_func=paddle.equal,
        )

    def test_equal_x_y_shard(self):
        if self._backend == "cpu":
            return
        self.test_binary_body(
            x_shape=[16, 32],
            y_shape=[16, 32],
            out_shape=[16, 32],
            x_specs=['x', None],
            y_specs=[None, 'x'],
            binary_func=paddle.equal,
        )

    def test_equal_x_y_shard_broadcast(self):
        self.test_binary_body(
            x_shape=[4, 16, 32],
            y_shape=[16, 32],
            out_shape=[4, 16, 32],
            x_specs=['x', None, None],
            y_specs=[None, None],
            binary_func=paddle.equal,
            rtol=1e-4,
        )

    def test_exp_x_shard(self):
        self.test_unary_body(
            x_shape=[4, 16],
            out_shape=[4, 16],
            x_specs=['x', None],
            unary_func=paddle.exp,
        )

    def test_rsqrt_x_shard(self):
        self.test_unary_body(
            x_shape=[4, 16],
            out_shape=[4, 16],
            x_specs=['x', None],
            unary_func=paddle.rsqrt,
        )

    def test_silu_x_shard(self):
        self.test_unary_body(
            x_shape=[4, 16],
            out_shape=[4, 16],
            x_specs=['x', None],
            unary_func=paddle.silu,
        )

    def run_test_case(self):
        if self._backend == "cpu":
            paddle.set_device("cpu")
        elif self._backend == "gpu":
            paddle.set_device("gpu:" + str(dist.get_rank()))
        else:
            raise ValueError("Only support cpu or gpu backend.")

        self.test_add_x_shard()
        self.test_add_x_shard_broadcast()
        self.test_add_x_y_shard()
        self.test_add_x_y_shard_broadcast()
        self.test_sub_x_shard()
        self.test_sub_x_y_shard_broadcast()
        self.test_square_x_shard()
        self.test_relu_x_shard()
        self.test_maximum_x_shard()
        self.test_maximum_x_shard_broadcast()
        self.test_maximum_x_y_shard()
        self.test_maximum_x_y_shard_broadcast()
        self.test_multiply_x_shard()
        self.test_multiply_x_shard_broadcast()
        self.test_multiply_x_y_shard()
        self.test_multiply_x_y_shard_broadcast()
        self.test_divide_x_shard()
        self.test_divide_x_shard_broadcast()
        self.test_divide_x_y_shard()
        self.test_divide_x_y_shard_broadcast()
        self.test_elementwise_pow_x_shard()
        self.test_elementwise_pow_x_shard_broadcast()
        self.test_elementwise_pow_x_y_shard()
        self.test_elementwise_pow_x_y_shard_broadcast()
        self.test_equal_x_shard()
        self.test_equal_x_shard_broadcast()
        self.test_equal_x_y_shard()
        self.test_equal_x_y_shard_broadcast()
        self.test_exp_x_shard()
        self.test_rsqrt_x_shard()
        self.test_silu_x_shard()


if __name__ == '__main__':
    TestElementwiseApiForSemiAutoParallel().run_test_case()
