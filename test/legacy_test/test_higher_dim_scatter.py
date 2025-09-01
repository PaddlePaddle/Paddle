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


@unittest.skipIf(
    not core.is_compiled_with_cuda(),
    "CPU scatter/gather kernel is not yet modified, coming soon and this skipping will be removed.",
)
class TestPutAlongAxisNonIncludeSelf2ndGrad(unittest.TestCase):
    """Test case from issue 72803"""

    def setUp(self):
        self.x = np.array(
            [
                [1.6947253, 1.7280283, -1.1000537, -1.7621638, -0.46924523],
                [-0.17813402, 0.9851728, 0.8784995, -0.35652128, 0.63679916],
                [-0.2506482, 0.46839848, 1.6940045, 1.2753638, -1.5601108],
                [-1.4223574, -0.30286825, -0.6940945, 0.4153872, -1.598482],
            ],
            dtype="float32",
        )
        self.indices = np.array(
            [
                [3, 2, 2, 2, 0],
                [1, 1, 3, 1, 3],
                [0, 0, 3, 2, 3],
                [0, 1, 2, 0, 3],
            ],
            dtype="int64",
        )
        self.values = np.array(
            [
                [-0.3371469, -2.3898945, -0.6047427, -0.18021728, 1.0270963],
                [-0.4792783, -0.06155855, -1.1657414, -0.22004248, -1.2116293],
                [-1.2325171, -1.2428453, -0.53471214, 0.64549965, 0.3991431],
                [-0.45945236, -0.2563897, -1.2712464, 1.7996459, -0.08381622],
            ],
            dtype="float32",
        )
        self.dout = np.array(
            [
                [-0.19797462, -0.98365456, 1.936407, -0.0050864, -1.0364918],
                [1.0826564, -2.1047552, 0.9298107, 0.6769417, 0.9323797],
                [-0.68968654, -0.5532966, 0.24068666, 0.5625817, 1.8991498],
                [0.84938127, -0.5345554, -0.6814333, -1.0064939, 2.419181],
            ],
            dtype="float32",
        )
        self.ddx = np.array(
            [
                [0.3573612, -0.6587053, -1.0527273, 0.7391721, -0.16440763],
                [-1.67882, -0.46170056, -0.81231886, 0.6644795, 1.0688623],
                [-1.3970909, 0.17792162, 0.35944283, -0.00945398, -1.8379706],
                [0.99883825, 0.47824964, -1.4997529, 0.80206966, -0.24591826],
            ],
            dtype="float32",
        )
        self.ddv = np.array(
            [
                [0.31652406, -0.41458955, -0.46466753, -0.23473991, 0.25190634],
                [-1.3948212, -0.84799731, 0.5940094, 0.46881115, 0.4054867],
                [-2.0037501, 0.087257907, 1.0091733, -0.002437128, 0.67401189],
                [-0.10354018, 0.51002628, -2.5794835, -1.7636456, -0.59410858],
            ],
            dtype="float32",
        )
        self.gt_result = np.array(
            [
                [-1.6919695, -1.2428453, -1.1000537, 1.7996459, 1.0270963],
                [-0.4792783, -0.31794825, 0.8784995, -0.22004248, 0.63679916],
                [-0.2506482, -2.3898945, -1.8759892, 0.46528238, -1.5601108],
                [-0.3371469, -0.30286825, -1.7004535, 0.4153872, -0.8963024],
            ],
            dtype="float32",
        )
        self.gt_dx = np.array(
            [
                [0.0, 0.0, 1.936407, 0.0, 0.0],
                [0.0, 0.0, 0.9298107, 0.0, 0.9323797],
                [-0.68968654, 0.0, 0.0, 0.0, 1.8991498],
                [0.0, -0.5345554, 0.0, -1.0064939, 0.0],
            ],
            dtype="float32",
        )
        self.gt_dv = np.array(
            [
                [0.84938127, -0.5532966, 0.24068666, 0.5625817, -1.0364918],
                [1.0826564, -2.1047552, -0.6814333, 0.6769417, 2.419181],
                [-0.19797462, -0.98365456, -0.6814333, 0.5625817, 2.419181],
                [-0.19797462, -2.1047552, 0.24068666, -0.0050864, 2.419181],
            ],
            dtype="float32",
        )
        self.gt_ddout = np.array(
            [
                [-2.1072903, 0.08725791, -1.0527273, -1.7636456, 0.25190634],
                [-1.3948212, -0.33797103, -0.81231886, 0.46881115, 1.0688623],
                [-1.3970909, -0.41458955, -3.044151, -0.23717704, -1.8379706],
                [0.31652406, 0.47824964, 1.6031827, 0.80206966, 0.48538995],
            ],
            dtype="float32",
        )

    def test_2nd_grad(self):
        x = paddle.to_tensor(self.x)
        x.stop_gradient = False
        include_self = False
        axis = 0

        indices = paddle.to_tensor(self.indices)

        values = paddle.to_tensor(self.values)
        values.stop_gradient = False

        out = paddle.put_along_axis(
            x,
            indices,
            values,
            axis,
            'add',
            include_self=include_self,
        )

        dout = paddle.to_tensor(self.dout)
        dout.stop_gradient = False

        dx, dv = paddle.grad(
            out,
            [x, values],
            dout,
            create_graph=True,
        )

        ddx = paddle.to_tensor(self.ddx)
        ddx.stop_gradient = False
        ddv = paddle.to_tensor(self.ddv)
        ddv.stop_gradient = False

        ddout = paddle.grad(
            [dx, dv],
            dout,
            [ddx, ddv],
        )[0]

        np.testing.assert_allclose(out.numpy(), self.gt_result, 1e-6, 1e-6)
        np.testing.assert_allclose(dx.numpy(), self.gt_dx, 1e-6, 1e-6)
        np.testing.assert_allclose(dv.numpy(), self.gt_dv, 1e-6, 1e-6)
        np.testing.assert_allclose(ddout.numpy(), self.gt_ddout, 1e-6, 1e-6)


if __name__ == '__main__':
    unittest.main()
