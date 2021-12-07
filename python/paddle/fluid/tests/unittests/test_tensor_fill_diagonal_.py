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

import paddle.fluid as fluid
import unittest
import numpy as np
import six
import paddle


class TensorFillDiagonal_Test(unittest.TestCase):
    def test_dim2_normal(self):
        expected_np = np.array(
            [[1, 2, 2], [2, 1, 2], [2, 2, 1]]).astype('float32')
        expected_grad = np.array(
            [[0, 1, 1], [1, 0, 1], [1, 1, 0]]).astype('float32')

        typelist = ['float32', 'float64', 'int32', 'int64']
        places = [fluid.CPUPlace()]
        if fluid.core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))

        for idx, p in enumerate(places):
            if idx == 0:
                paddle.set_device('cpu')
            else:
                paddle.set_device('gpu')
            for dtype in typelist:
                x = paddle.ones((3, 3), dtype=dtype)
                x.stop_gradient = False
                y = x * 2
                y.fill_diagonal_(1, offset=0, wrap=True)
                loss = y.sum()
                loss.backward()

                self.assertEqual(
                    (y.numpy().astype('float32') == expected_np).all(), True)
                self.assertEqual(
                    (y.grad.numpy().astype('float32') == expected_grad).all(),
                    True)

    def test_bool(self):
        expected_np = np.array(
            [[False, True, True], [True, False, True], [True, True, False]])

        typelist = ['bool']
        places = [fluid.CPUPlace()]
        if fluid.core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))

        for idx, p in enumerate(places):
            if idx == 0:
                paddle.set_device('cpu')
            else:
                paddle.set_device('gpu')
            for dtype in typelist:
                x = paddle.ones((3, 3), dtype=dtype)
                x.stop_gradient = True
                x.fill_diagonal_(0, offset=0, wrap=True)

                self.assertEqual((x.numpy() == expected_np).all(), True)

    def test_dim2_unnormal_wrap(self):
        expected_np = np.array([[1, 2, 2], [2, 1, 2], [2, 2, 1], [2, 2, 2],
                                [1, 2, 2], [2, 1, 2],
                                [2, 2, 1]]).astype('float32')
        expected_grad = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1],
                                  [0, 1, 1], [1, 0, 1],
                                  [1, 1, 0]]).astype('float32')

        typelist = ['float32', 'float64', 'int32', 'int64']
        places = [fluid.CPUPlace()]
        if fluid.core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))

        for idx, p in enumerate(places):
            if idx == 0:
                paddle.set_device('cpu')
            else:
                paddle.set_device('gpu')
            for dtype in typelist:
                x = paddle.ones((7, 3), dtype=dtype)
                x.stop_gradient = False
                y = x * 2
                y.fill_diagonal_(1, offset=0, wrap=True)
                loss = y.sum()
                loss.backward()

                self.assertEqual(
                    (y.numpy().astype('float32') == expected_np).all(), True)
                self.assertEqual(
                    (y.grad.numpy().astype('float32') == expected_grad).all(),
                    True)

    def test_dim2_unnormal_unwrap(self):
        expected_np = np.array([[1, 2, 2], [2, 1, 2], [2, 2, 1], [2, 2, 2],
                                [2, 2, 2], [2, 2, 2],
                                [2, 2, 2]]).astype('float32')
        expected_grad = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1],
                                  [1, 1, 1], [1, 1, 1],
                                  [1, 1, 1]]).astype('float32')

        typelist = ['float32', 'float64', 'int32', 'int64']
        places = [fluid.CPUPlace()]
        if fluid.core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))

        for idx, p in enumerate(places):
            if idx == 0:
                paddle.set_device('cpu')
            else:
                paddle.set_device('gpu')
            for dtype in typelist:
                x = paddle.ones((7, 3), dtype=dtype)
                x.stop_gradient = False
                y = x * 2
                y.fill_diagonal_(1, offset=0, wrap=False)
                loss = y.sum()
                loss.backward()

                self.assertEqual(
                    (y.numpy().astype('float32') == expected_np).all(), True)
                self.assertEqual(
                    (y.grad.numpy().astype('float32') == expected_grad).all(),
                    True)

    def test_dim_larger2_normal(self):
        expected_np = np.array([[[1, 2, 2], [2, 2, 2], [2, 2, 2]], [[2, 2, 2], [
            2, 1, 2
        ], [2, 2, 2]], [[2, 2, 2], [2, 2, 2], [2, 2, 1]]]).astype('float32')
        expected_grad = np.array(
            [[[0, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 0, 1],
                                                 [1, 1, 1]],
             [[1, 1, 1], [1, 1, 1], [1, 1, 0]]]).astype('float32')

        typelist = ['float32', 'float64', 'int32', 'int64']
        places = [fluid.CPUPlace()]
        if fluid.core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))

        for idx, p in enumerate(places):
            if idx == 0:
                paddle.set_device('cpu')
            else:
                paddle.set_device('gpu')
            for dtype in typelist:
                x = paddle.ones((3, 3, 3), dtype=dtype)
                x.stop_gradient = False
                y = x * 2
                y.fill_diagonal_(1, offset=0, wrap=True)
                loss = y.sum()
                loss.backward()

                self.assertEqual(
                    (y.numpy().astype('float32') == expected_np).all(), True)
                self.assertEqual(
                    (y.grad.numpy().astype('float32') == expected_grad).all(),
                    True)


if __name__ == '__main__':
    unittest.main()
