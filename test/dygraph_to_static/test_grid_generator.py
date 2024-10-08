# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
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
    enable_to_static_guard,
)

import paddle
from paddle import ParamAttr, nn

np.random.seed(2020)
paddle.seed(2020)


class GridGenerator(nn.Layer):
    def __init__(self, in_channels, num_fiducial):
        super().__init__()
        self.eps = 1e-6
        self.F = num_fiducial

        initializer = nn.initializer.Constant(value=0.0)
        param_attr = ParamAttr(learning_rate=0.0, initializer=initializer)
        bias_attr = ParamAttr(learning_rate=0.0, initializer=initializer)
        self.fc = nn.Linear(
            in_channels, 6, weight_attr=param_attr, bias_attr=bias_attr
        )

    def forward(self, batch_C_prime, I_r_size):
        """
        Generate the grid for the grid_sampler.
        Args:
            batch_C_prime: the matrix of the geometric transformation
            I_r_size: the shape of the input image
        Return:
            batch_P_prime: the grid for the grid_sampler
        """
        C = self.build_C_paddle()
        return C

    def build_C_paddle(self):
        """Return coordinates of fiducial points in I_r; C"""
        F = self.F
        ctrl_pts_x = paddle.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = -1 * paddle.ones([int(F / 2)])
        ctrl_pts_y_bottom = paddle.ones([int(F / 2)])
        ctrl_pts_top = paddle.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = paddle.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        C = paddle.concat([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return C

    def build_P_paddle(self, I_r_size):
        I_r_width, I_r_height = I_r_size
        I_r_grid_x = paddle.divide(
            (paddle.arange(-I_r_width, I_r_width, 2).astype('float32') + 1.0),
            paddle.to_tensor(I_r_width).astype('float32'),
        )
        I_r_grid_y = paddle.divide(
            (paddle.arange(-I_r_height, I_r_height, 2).astype('float32') + 1.0),
            paddle.to_tensor(I_r_height).astype('float32'),
        )
        P = paddle.stack(paddle.meshgrid(I_r_grid_x, I_r_grid_y), axis=2)
        P = paddle.transpose(P, perm=[1, 0, 2])
        return P.reshape([-1, 2])

    def build_inv_delta_C_paddle(self, C):
        """Return inv_delta_C which is needed to calculate T"""
        F = self.F
        hat_C = paddle.zeros((F, F), dtype='float32')
        for i in range(0, F):
            for j in range(i, F):
                if i == j:
                    hat_C[i, j] = 1
                else:
                    r = paddle.norm(C[i] - C[j])
                    hat_C[i, j] = r
                    hat_C[j, i] = r
        hat_C = (hat_C**2) * paddle.log(hat_C)
        delta_C = paddle.concat(
            [
                paddle.concat([paddle.ones((F, 1)), C, hat_C], axis=1),
                paddle.concat(
                    [paddle.zeros((2, 3)), paddle.transpose(C, perm=[1, 0])],
                    axis=1,
                ),
                paddle.concat(
                    [paddle.zeros((1, 3)), paddle.ones((1, F))], axis=1
                ),
            ],
            axis=0,
        )
        inv_delta_C = paddle.inverse(delta_C)
        return inv_delta_C

    def build_P_hat_paddle(self, C, P):
        F = self.F
        eps = self.eps
        n = P.shape[0]
        P_tile = paddle.tile(paddle.unsqueeze(P, axis=1), (1, F, 1))
        C_tile = paddle.unsqueeze(C, axis=0)
        P_diff = P_tile - C_tile
        rbf_norm = paddle.norm(P_diff, p=2, axis=2, keepdim=False)

        rbf = paddle.multiply(
            paddle.square(rbf_norm), paddle.log(rbf_norm + eps)
        )
        P_hat = paddle.concat([paddle.ones((n, 1)), P, rbf], axis=1)
        return P_hat

    def get_expand_tensor(self, batch_C_prime):
        B, H, C = batch_C_prime.shape
        batch_C_prime = batch_C_prime.reshape([B, H * C])
        batch_C_ex_part_tensor = self.fc(batch_C_prime)
        batch_C_ex_part_tensor = batch_C_ex_part_tensor.reshape([-1, 3, 2])
        return batch_C_ex_part_tensor


class TestGridGenerator(Dy2StTestBase):
    def setUp(self):
        self.x = paddle.uniform(shape=[1, 20, 2], dtype='float32')

    def _run(self, to_static):
        with enable_to_static_guard(to_static):
            net = paddle.jit.to_static(
                GridGenerator(40, 20),
                input_spec=[
                    paddle.static.InputSpec(
                        shape=[None, 3, 32, 100], dtype='float32'
                    ),
                ],
            )
            ret = net(self.x, [32, 100])
        return ret.numpy()

    def test_to_static(self):
        st_out = self._run(to_static=True)
        dy_out = self._run(to_static=False)
        np.testing.assert_allclose(st_out, dy_out)


if __name__ == '__main__':
    unittest.main()
