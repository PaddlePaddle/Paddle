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
import utils

import paddle


class TestGatherNd(unittest.TestCase):
    # Note that GatherNd is also used in index_put, so we can test it by using index_put.
    def eval(self, dy_compute, inputs, input_spec=None):
        dy_out = dy_compute(*inputs)

        static_compute = utils.apply_to_static(
            dy_compute, use_cinn=True, input_spec=None
        )
        st_out = static_compute(*inputs)

        for a, b in zip(
            paddle.utils.flatten(dy_out), paddle.utils.flatten(st_out)
        ):
            numpy.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-6)

    @staticmethod
    def get_input(x_shape, indices_shape, value_shape, has_negative_index=True):
        n_indices = indices_shape[0]
        index_dim_size = indices_shape[1] if len(indices_shape) > 1 else 1

        x_pd = paddle.randn(x_shape)
        x_pd.stop_gradient = False

        indices_pd = tuple(
            [
                paddle.randint(
                    -x_shape[i] if has_negative_index else 0,
                    x_shape[i],
                    [n_indices],
                )
                for i in range(max(index_dim_size, 1))
            ]
        )
        value_pd = paddle.randn(value_shape)
        value_pd.stop_gradient = False

        dout_pd = paddle.randn(x_shape)
        dout_pd.stop_gradient = False
        return x_pd, indices_pd, value_pd, dout_pd

    @staticmethod
    def get_input_spec(indice_dim):
        return [
            paddle.static.InputSpec(shape=[-1, -1], dtype="float32"),
            tuple(
                paddle.static.InputSpec(shape=[-1], dtype="int64")
                for _ in range(indice_dim)
            ),
            paddle.static.InputSpec(shape=[-1, -1], dtype="float32"),
            paddle.static.InputSpec(shape=[-1, -1], dtype="float32"),
        ]

    @staticmethod
    def index_put_grad(x, indices, v, dy):
        y = paddle.index_put(x, indices, v, True)
        return paddle.grad(y, [x, v], dy)

    def test_index_put_grad_non_negative_index(self):
        x_pd, indices_pd, value_pd, dout_pd = self.get_input(
            [12, 13, 14], [88, 2], [88, 14], False
        )

        self.eval(
            TestGatherNd.index_put_grad,
            [x_pd, indices_pd, value_pd, dout_pd],
            input_spec=self.get_input_spec(2),
        )

    def test_index_put_grad_negative_index_1(self):
        x_pd, indices_pd, value_pd, dout_pd = self.get_input(
            [12, 13, 14], [88, 1], [88, 13, 14]
        )

        self.eval(
            TestGatherNd.index_put_grad,
            [x_pd, indices_pd, value_pd, dout_pd],
            input_spec=self.get_input_spec(1),
        )

    def test_index_put_grad_negative_index_2(self):
        x_pd, indices_pd, value_pd, dout_pd = self.get_input(
            [16, 16], [20, 2], [20]
        )

        self.eval(
            TestGatherNd.index_put_grad,
            [x_pd, indices_pd, value_pd, dout_pd],
            input_spec=self.get_input_spec(2),
        )

    def test_gather_nd_fusion(self):
        x_pd = paddle.randn([256, 128])
        y_pd = paddle.randn_like(x_pd)
        z_pd = paddle.randn([100])
        indices_pd = paddle.randint(-128, 128, [100, 2])

        def func(x, y, z, indices):
            return paddle.gather_nd(x * y, indices) + z

        self.eval(func, [x_pd, y_pd, z_pd, indices_pd])


if __name__ == "__main__":
    unittest.main()
