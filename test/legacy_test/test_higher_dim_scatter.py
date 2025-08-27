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

import numpy as np

import paddle
from paddle import core


@unittest.skipIf(
    not core.is_compiled_with_cuda(),
    "CPU scatter/gather kernel is not yet modified, coming soon and this skipping will be removed.",
)
class TestNonBroadcastableMismatchedShapeCase(unittest.TestCase):
    """Unittest from PyTorch comparison and handcrafted backward result
    Note that this unit test might fail, if you modify the implementation
    of scatter and gather kernel, especially the ordering of atomic writes

    So make sure you know what you are doing, otherwise
    you may need to update this unittest.
    """

    def setUp(self):
        self.input = paddle.to_tensor(
            [
                [
                    [
                        [1.9693925, 2.2913685],
                        [-0.19461553, 0.298859],
                        [-0.86006254, 0.28243607],
                    ],
                    [
                        [-0.09577879, -0.10506158],
                        [-0.12375893, 1.4438118],
                        [-0.66273206, 1.0404967],
                    ],
                ],
                [
                    [
                        [0.29458013, 0.51647896],
                        [0.79423386, -1.5084593],
                        [0.405428, -0.8155419],
                    ],
                    [
                        [0.27907062, 0.70933336],
                        [-1.2590513, 0.7363407],
                        [1.078117, -0.03632839],
                    ],
                ],
            ],
            dtype='float32',
            stop_gradient=False,
        )
        self.index = paddle.to_tensor(
            [[[[0], [1]]], [[[1], [0]]]], dtype='int64', stop_gradient=True
        )
        self.src = paddle.to_tensor(
            [
                [
                    [[-2.1342657], [-0.6801669], [-0.741744]],
                    [[-0.15918107], [1.5543042], [-0.35116914]],
                ],
                [
                    [[0.39571938], [0.5322498], [-0.35833976]],
                    [[1.3826214], [0.6314196], [0.891596]],
                ],
            ],
            dtype='float32',
            stop_gradient=False,
        )
        self.no_grad = False
        self.dim = 2
        self.include_self = True

    def test_no_grad_add(self):
        self.input.clear_grad()
        self.src.clear_grad()
        result = paddle.put_along_axis(
            self.input,
            indices=self.index,
            values=self.src,
            axis=self.dim,
            reduce='add',
            include_self=self.include_self,
            broadcast=False,
        )
        gt_result = np.array(
            [
                [
                    [
                        [-0.16487312, 2.2913685],
                        [-0.87478244, 0.298859],
                        [-0.86006254, 0.28243607],
                    ],
                    [
                        [-0.09577879, -0.10506158],
                        [-0.12375893, 1.4438118],
                        [-0.66273206, 1.0404967],
                    ],
                ],
                [
                    [
                        [0.8268299, 0.51647896],
                        [1.1899532, -1.5084593],
                        [0.405428, -0.8155419],
                    ],
                    [
                        [0.27907062, 0.70933336],
                        [-1.2590513, 0.7363407],
                        [1.078117, -0.03632839],
                    ],
                ],
            ],
            dtype='float32',
        )
        np.testing.assert_allclose(
            result.numpy(), gt_result, rtol=1e-6, atol=1e-6
        )

    def test_with_grad_assign(self):
        self.input.clear_grad()
        self.src.clear_grad()
        result = paddle.put_along_axis(
            self.input,
            indices=self.index,
            values=self.src,
            axis=self.dim,
            reduce='assign',
            include_self=self.include_self,
            broadcast=False,
        )
        gt_result = np.array(
            [
                [
                    [
                        [-2.1342657, 2.2913685],
                        [-0.6801669, 0.298859],
                        [-0.86006254, 0.28243607],
                    ],
                    [
                        [-0.09577879, -0.10506158],
                        [-0.12375893, 1.4438118],
                        [-0.66273206, 1.0404967],
                    ],
                ],
                [
                    [
                        [0.5322498, 0.51647896],
                        [0.39571938, -1.5084593],
                        [0.405428, -0.8155419],
                    ],
                    [
                        [0.27907062, 0.70933336],
                        [-1.2590513, 0.7363407],
                        [1.078117, -0.03632839],
                    ],
                ],
            ],
            dtype='float32',
        )
        np.testing.assert_allclose(
            result.numpy(), gt_result, rtol=1e-6, atol=1e-6
        )

        result.backward()
        gt_input_grad = np.array(
            [
                [
                    [[0.0, 1.0], [0.0, 1.0], [1.0, 1.0]],
                    [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                ],
                [
                    [[0.0, 1.0], [0.0, 1.0], [1.0, 1.0]],
                    [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                ],
            ],
            dtype='float32',
        )
        gt_src_grad = np.array(
            [[[[1.0], [1.0]]], [[[1.0], [1.0]]]], dtype='float32'
        )
        np.testing.assert_allclose(
            self.input.grad.numpy(), gt_input_grad, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            self.src.grad.numpy(), gt_src_grad, rtol=1e-6, atol=1e-6
        )

    def test_no_grad_mul(self):
        self.input.clear_grad()
        self.src.clear_grad()
        result = paddle.put_along_axis(
            self.input,
            indices=self.index,
            values=self.src,
            axis=self.dim,
            reduce='mul',
            include_self=self.include_self,
            broadcast=False,
        )
        gt_result = np.array(
            [
                [
                    [
                        [-4.203207, 2.2913685],
                        [0.13237104, 0.298859],
                        [-0.86006254, 0.28243607],
                    ],
                    [
                        [-0.09577879, -0.10506158],
                        [-0.12375893, 1.4438118],
                        [-0.66273206, 1.0404967],
                    ],
                ],
                [
                    [
                        [0.15679021, 0.51647896],
                        [0.31429374, -1.5084593],
                        [0.405428, -0.8155419],
                    ],
                    [
                        [0.27907062, 0.70933336],
                        [-1.2590513, 0.7363407],
                        [1.078117, -0.03632839],
                    ],
                ],
            ],
            dtype='float32',
        )
        np.testing.assert_allclose(
            result.numpy(), gt_result, rtol=1e-6, atol=1e-6
        )

    def test_with_grad_amin(self):
        self.input.clear_grad()
        self.src.clear_grad()
        result = paddle.put_along_axis(
            self.input,
            indices=self.index,
            values=self.src,
            axis=self.dim,
            reduce='amin',
            include_self=self.include_self,
            broadcast=False,
        )
        gt_result = np.array(
            [
                [
                    [
                        [-2.1342657, 2.2913685],
                        [-0.6801669, 0.298859],
                        [-0.86006254, 0.28243607],
                    ],
                    [
                        [-0.09577879, -0.10506158],
                        [-0.12375893, 1.4438118],
                        [-0.66273206, 1.0404967],
                    ],
                ],
                [
                    [
                        [0.29458013, 0.51647896],
                        [0.39571938, -1.5084593],
                        [0.405428, -0.8155419],
                    ],
                    [
                        [0.27907062, 0.70933336],
                        [-1.2590513, 0.7363407],
                        [1.078117, -0.03632839],
                    ],
                ],
            ],
            dtype='float32',
        )
        np.testing.assert_allclose(
            result.numpy(), gt_result, rtol=1e-6, atol=1e-6
        )

        result.backward()
        gt_input_grad = np.array(
            [
                [
                    [[0.0, 1.0], [0.0, 1.0], [1.0, 1.0]],
                    [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                ],
                [
                    [[1.0, 1.0], [0.0, 1.0], [1.0, 1.0]],
                    [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                ],
            ],
            dtype='float32',
        )
        gt_src_grad = np.array(
            [[[[1.0], [1.0]]], [[[0.0], [0.0]]]], dtype='float32'
        )
        np.testing.assert_allclose(
            self.input.grad.numpy(), gt_input_grad, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            self.src.grad.numpy(), gt_src_grad, rtol=1e-6, atol=1e-6
        )

    def test_with_grad_amax(self):
        self.input.clear_grad()
        self.src.clear_grad()
        result = paddle.put_along_axis(
            self.input,
            indices=self.index,
            values=self.src,
            axis=self.dim,
            reduce='amax',
            include_self=self.include_self,
            broadcast=False,
        )
        gt_result = np.array(
            [
                [
                    [
                        [1.9693925, 2.2913685],
                        [-0.19461553, 0.298859],
                        [-0.86006254, 0.28243607],
                    ],
                    [
                        [-0.09577879, -0.10506158],
                        [-0.12375893, 1.4438118],
                        [-0.66273206, 1.0404967],
                    ],
                ],
                [
                    [
                        [0.5322498, 0.51647896],
                        [0.79423386, -1.5084593],
                        [0.405428, -0.8155419],
                    ],
                    [
                        [0.27907062, 0.70933336],
                        [-1.2590513, 0.7363407],
                        [1.078117, -0.03632839],
                    ],
                ],
            ],
            dtype='float32',
        )
        np.testing.assert_allclose(
            result.numpy(), gt_result, rtol=1e-6, atol=1e-6
        )

        result.backward()
        gt_input_grad = np.array(
            [
                [
                    [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                    [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                ],
                [
                    [[0.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                    [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                ],
            ],
            dtype='float32',
        )
        gt_src_grad = np.array(
            [[[[0.0], [0.0]]], [[[0.0], [0.0]]]], dtype='float32'
        )
        np.testing.assert_allclose(
            self.input.grad.numpy(), gt_input_grad, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            self.src.grad.numpy(), gt_src_grad, rtol=1e-6, atol=1e-6
        )

    def test_no_grad_mean(self):
        self.input.clear_grad()
        self.src.clear_grad()
        result = paddle.put_along_axis(
            self.input,
            indices=self.index,
            values=self.src,
            axis=self.dim,
            reduce='mean',
            include_self=self.include_self,
            broadcast=False,
        )
        gt_result = np.array(
            [
                [
                    [
                        [-0.08243656, 2.2913685],
                        [-0.43739122, 0.298859],
                        [-0.86006254, 0.28243607],
                    ],
                    [
                        [-0.09577879, -0.10506158],
                        [-0.12375893, 1.4438118],
                        [-0.66273206, 1.0404967],
                    ],
                ],
                [
                    [
                        [0.41341496, 0.51647896],
                        [0.5949766, -1.5084593],
                        [0.405428, -0.8155419],
                    ],
                    [
                        [0.27907062, 0.70933336],
                        [-1.2590513, 0.7363407],
                        [1.078117, -0.03632839],
                    ],
                ],
            ],
            dtype='float32',
        )
        np.testing.assert_allclose(
            result.numpy(), gt_result, rtol=1e-6, atol=1e-6
        )


if __name__ == '__main__':
    unittest.main()
