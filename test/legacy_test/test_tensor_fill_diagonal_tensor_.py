# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

import paddle
from paddle import base


class TensorFillDiagTensor_Test(unittest.TestCase):
    def setUp(self):
        self.typelist = ['float32', 'float64', 'int32', 'int64']
        self.places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not base.core.is_compiled_with_cuda()
        ):
            self.places.append(base.CPUPlace())
        if base.core.is_compiled_with_cuda():
            self.places.append(base.CUDAPlace(0))

    def test_dim2(self):
        expected_np = np.array(
            [[1, 2, 2], [2, 1, 2], [2, 2, 1], [2, 2, 2]]
        ).astype('float32')
        expected_grad = np.array(
            [[0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
        ).astype('float32')

        for idx, p in enumerate(self.places):
            if idx == 0:
                paddle.set_device('cpu')
            else:
                paddle.set_device('gpu')
            for dtype in self.typelist:
                v = paddle.ones((3,), dtype=dtype)
                var = np.random.random() + 1
                x = paddle.ones((4, 3), dtype=dtype)
                x.stop_gradient = False
                y = x * 2
                y.retain_grads()
                y.fill_diagonal_tensor_(v, offset=0, dim1=0, dim2=1)
                loss = y.sum()
                loss.backward()

                self.assertEqual(
                    (y.numpy().astype('float32') == expected_np).all(), True
                )
                self.assertEqual(
                    (y.grad.numpy().astype('float32') == expected_grad).all(),
                    True,
                )

    def test_dim2_offset_1(self):
        expected_np = np.array(
            [[2, 2, 2], [1, 2, 2], [2, 1, 2], [2, 2, 1]]
        ).astype('float32')
        expected_grad = np.array(
            [[1, 1, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
        ).astype('float32')

        for idx, p in enumerate(self.places):
            if idx == 0:
                paddle.set_device('cpu')
            else:
                paddle.set_device('gpu')
            for dtype in self.typelist:
                v = paddle.ones((3,), dtype=dtype)
                var = np.random.random() + 1
                x = paddle.ones((4, 3), dtype=dtype)
                x.stop_gradient = False
                y = x * 2
                y.retain_grads()
                y.fill_diagonal_tensor_(v, offset=-1, dim1=0, dim2=1)
                loss = y.sum()
                loss.backward()

                self.assertEqual(
                    (y.numpy().astype('float32') == expected_np).all(), True
                )
                self.assertEqual(
                    (y.grad.numpy().astype('float32') == expected_grad).all(),
                    True,
                )

    def test_dim2_offset1(self):
        expected_np = np.array(
            [[2, 1, 2], [2, 2, 1], [2, 2, 2], [2, 2, 2]]
        ).astype('float32')
        expected_grad = np.array(
            [[1, 0, 1], [1, 1, 0], [1, 1, 1], [1, 1, 1]]
        ).astype('float32')

        for idx, p in enumerate(self.places):
            if idx == 0:
                paddle.set_device('cpu')
            else:
                paddle.set_device('gpu')
            for dtype in self.typelist:
                v = paddle.ones((2,), dtype=dtype)
                var = np.random.random() + 1
                x = paddle.ones((4, 3), dtype=dtype)
                x.stop_gradient = False
                y = x * 2
                y.retain_grads()
                y.fill_diagonal_tensor_(v, offset=1, dim1=0, dim2=1)
                loss = y.sum()
                loss.backward()

                self.assertEqual(
                    (y.numpy().astype('float32') == expected_np).all(), True
                )
                self.assertEqual(
                    (y.grad.numpy().astype('float32') == expected_grad).all(),
                    True,
                )

    def test_dim4(self):
        expected_np = np.array(
            [
                [
                    [[0, 3], [2, 2], [2, 2]],
                    [[2, 2], [1, 4], [2, 2]],
                    [[2, 2], [2, 2], [2, 5]],
                    [[2, 2], [2, 2], [2, 2]],
                ],
                [
                    [[6, 9], [2, 2], [2, 2]],
                    [[2, 2], [7, 10], [2, 2]],
                    [[2, 2], [2, 2], [8, 11]],
                    [[2, 2], [2, 2], [2, 2]],
                ],
            ]
        ).astype('float32')
        expected_grad = np.array(
            [
                [
                    [[0, 0], [1, 1], [1, 1]],
                    [[1, 1], [0, 0], [1, 1]],
                    [[1, 1], [1, 1], [0, 0]],
                    [[1, 1], [1, 1], [1, 1]],
                ],
                [
                    [[0, 0], [1, 1], [1, 1]],
                    [[1, 1], [0, 0], [1, 1]],
                    [[1, 1], [1, 1], [0, 0]],
                    [[1, 1], [1, 1], [1, 1]],
                ],
            ]
        ).astype('float32')

        for idx, p in enumerate(self.places):
            if idx == 0:
                paddle.set_device('cpu')
            else:
                paddle.set_device('gpu')
            for dtype in self.typelist:
                v = paddle.to_tensor(
                    np.arange(12).reshape(2, 2, 3), dtype=dtype
                )
                var = np.random.random() + 1
                x = paddle.ones((2, 4, 3, 2), dtype=dtype)
                x.stop_gradient = False
                y = x * 2
                y.retain_grads()
                y.fill_diagonal_tensor_(v, offset=0, dim1=1, dim2=2)
                loss = y.sum()
                loss.backward()

                self.assertEqual(
                    (y.numpy().astype('float32') == expected_np).all(), True
                )
                self.assertEqual(
                    (y.grad.numpy().astype('float32') == expected_grad).all(),
                    True,
                )

    def test_largedim(self):
        # large dim only test on gpu because the cpu version is too slow for ci test, and the memory is limited
        if len(self.places) > 1:
            bsdim = 1024
            fsdim = 128
            paddle.set_device('gpu')
            for dtype in self.typelist:
                v = paddle.arange(bsdim * fsdim, dtype=dtype).reshape(
                    (bsdim, fsdim)
                )
                y = paddle.ones((bsdim, fsdim, fsdim), dtype=dtype)
                y.stop_gradient = False
                y = y * 2
                y.retain_grads()
                y.fill_diagonal_tensor_(v, offset=0, dim1=1, dim2=2)
                loss = y.sum()
                loss.backward()

                expected_pred = v - 2
                expected_pred = paddle.diag_embed(expected_pred) + 2
                expected_grad = paddle.ones(v.shape, dtype=dtype) - 2
                expected_grad = paddle.diag_embed(expected_grad) + 1

                self.assertEqual((y == expected_pred).all(), True)
                self.assertEqual((y.grad == expected_grad).all(), True)


if __name__ == '__main__':
    unittest.main()
