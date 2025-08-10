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


class TestCompatMinMax(unittest.TestCase):
    def setUp(self):
        """Make sure we are in a dynamic graph env"""
        paddle.disable_static()

    def test_case1_simple_reduce_all(self):
        data = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype='float32')
        min_val = paddle.compat.min(data)
        max_val = paddle.compat.max(data)

        self.assertAlmostEqual(min_val.item(), 1.0)
        self.assertAlmostEqual(max_val.item(), 4.0)

        data = paddle.to_tensor(
            [[1.0, 1.0], [2.0, 3.0]], dtype='float32', stop_gradient=False
        )
        min_val = paddle.compat.min(data)
        min_val.backward()

        expected_grad = np.array([[0.5, 0.5], [0.0, 0.0]])
        np.testing.assert_allclose(data.grad.numpy(), expected_grad)

    def test_case2_reduce_dim(self):
        """Test dim/keepdim"""
        data = paddle.to_tensor(
            [[[5, 8], [2, 1]], [[7, 3], [9, 6]]], dtype='float32'
        )

        min_result = paddle.compat.min(data, dim=1)
        self.assertEqual(min_result.values.shape, [2, 2])
        np.testing.assert_array_equal(
            min_result.values.numpy(), np.array([[2, 1], [7, 3]])
        )
        np.testing.assert_array_equal(
            min_result.indices.numpy(), np.array([[1, 1], [0, 0]])
        )

        max_result = paddle.compat.max(data, dim=2)
        self.assertEqual(max_result.values.shape, [2, 2])
        np.testing.assert_array_equal(
            max_result.values.numpy(), np.array([[8, 2], [7, 9]])
        )
        np.testing.assert_array_equal(
            max_result.indices.numpy(), np.array([[1, 0], [0, 0]])
        )

        min_result_keep = paddle.compat.min(data, dim=0, keepdim=True)
        self.assertEqual(min_result_keep.values.shape, [1, 2, 2])
        np.testing.assert_array_equal(
            min_result_keep.values.numpy(), np.array([[[5, 3], [2, 1]]])
        )

        min_result_neg = paddle.compat.min(data, dim=-2)
        np.testing.assert_array_equal(
            min_result_neg.values.numpy(), min_result.values.numpy()
        )

    def test_case2_grad(self):
        data = paddle.to_tensor(
            [[[1.0, 2.0], [1.0, 3.0]], [[4.0, 1.0], [5.0, 1.0]]],
            dtype='float32',
            stop_gradient=False,
        )
        y = data * 2

        min_result = paddle.compat.min(y, dim=2)
        min_result.values.backward()

        expected_grad = np.array(
            [[[2.0, 0.0], [2.0, 0.0]], [[0.0, 2.0], [0.0, 2.0]]]
        )
        np.testing.assert_allclose(data.grad.numpy(), expected_grad, atol=1e-6)

        data.clear_grad()
        y = data * data
        min_result = paddle.compat.min(y, dim=1)
        min_result[0].backward()
        expected_grad = np.array(
            [[[2.0, 4.0], [0.0, 0.0]], [[8.0, 2.0], [0.0, 0.0]]]
        )
        np.testing.assert_allclose(data.grad.numpy(), expected_grad, atol=1e-6)

    def test_case3_elementwise(self):
        """minimum/maximum"""
        x = paddle.to_tensor([[1, 5], [4, 2]], dtype='float32')
        y = paddle.to_tensor([[3, 2], [1, 6]], dtype='float32')

        min_result = paddle.compat.min(x, y)
        np.testing.assert_array_equal(
            min_result.numpy(), np.array([[1, 2], [1, 2]])
        )

        max_result = paddle.compat.max(x, y)
        np.testing.assert_array_equal(
            max_result.numpy(), np.array([[3, 5], [4, 6]])
        )

        z = paddle.to_tensor([3, 4], dtype='float32')
        broadcast_min = paddle.compat.min(x, z)
        np.testing.assert_array_equal(
            broadcast_min.numpy(), np.array([[1, 4], [3, 2]])
        )

    def test_case3_grad(self):
        x = paddle.to_tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=paddle.float32, stop_gradient=False
        )
        y = paddle.to_tensor(
            [[0.5, 2.5], [2.0, 3.5]], dtype=paddle.float32, stop_gradient=False
        )

        min_val = paddle.compat.min(x, y)
        min_val.backward()

        expected_x_grad = np.array([[0.0, 1.0], [0.0, 0.0]])
        np.testing.assert_allclose(x.grad.numpy(), expected_x_grad)

        expected_y_grad = np.array([[1.0, 0.0], [1.0, 1.0]])
        np.testing.assert_allclose(y.grad.numpy(), expected_y_grad)

    def test_edge_cases(self):
        """Edge cases test"""
        # uniform distributed gradient
        uniform_data = paddle.ones([2, 3], dtype='float64')
        uniform_data.stop_gradient = False
        min_val = paddle.compat.min(uniform_data)
        min_val.sum().backward()
        # uniformly distributed (amin)
        expected_grad = np.full((2, 3), 1.0 / 6.0)
        np.testing.assert_allclose(uniform_data.grad.numpy(), expected_grad)

        uniform_data.clear_grad()
        min_val = paddle.compat.min(uniform_data, 0)
        min_val.values.sum().backward()
        # take_along_axis like gradient behavior
        expected_grad = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
        np.testing.assert_allclose(uniform_data.grad.numpy(), expected_grad)

        # 0-dim tensor
        dim0_tensor = paddle.to_tensor(2, dtype='float32')
        max_val = paddle.compat.max(dim0_tensor)
        np.testing.assert_allclose(
            max_val.numpy(), np.array(2.0, dtype=np.float32)
        )

        # 1-dim tensor
        dim1_tensor = paddle.to_tensor([1], dtype='uint8')
        max_val = paddle.compat.max(dim1_tensor, dim=-1, keepdim=True)
        np.testing.assert_array_equal(
            max_val[0].numpy(), np.array([1], dtype=np.uint8)
        )
        np.testing.assert_array_equal(
            max_val[1].numpy(), np.array([0], dtype=np.int64)
        )

    def test_compare_with_index_ops_to_origin(self):
        dtypes = ['float32', 'float64', 'int32', 'int64', 'uint8']
        cpu_reject_types = {'int16', 'bfloat16', 'float16'}

        for i, dtype in enumerate(dtypes):
            data = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype=dtype)
            # `bfloat16` and `float16` are rejected on CPU
            if not data.place.is_gpu_place() and dtype in cpu_reject_types:
                continue
            min_vals_inds = paddle.compat.min(data, dim=0)
            self.assertEqual(min_vals_inds.values.dtype, data.dtype)
            self.assertEqual(min_vals_inds.indices.dtype, paddle.int64)

            origin_indices = paddle.argmin(data, axis=0, dtype="int64")
            if dtype != 'uint8':
                origin_values = paddle.min(data, axis=0)
            else:
                origin_values = paddle.take_along_axis(
                    data, origin_indices.unsqueeze(0), axis=0
                )
                origin_values.squeeze_(axis=0)
            if i < 4:  # floating point
                np.testing.assert_allclose(
                    min_vals_inds.values.numpy(), origin_values.numpy()
                )
            else:
                np.testing.assert_array_equal(
                    min_vals_inds.values.numpy(), origin_values.numpy()
                )
            np.testing.assert_array_equal(
                min_vals_inds[1].numpy(), origin_indices.numpy()
            )

    def test_error_handling(self):
        """Test whether correct exception will be thrown. Skip error messages (some of them are long)"""

        err_msg1 = (
            "Tensors with integral type: 'paddle.int32' should stop gradient."
        )
        err_msg2 = (
            "paddle.min() received unexpected keyword arguments 'input', 'dim'. "
            "\nDid you mean to use paddle.compat.min() instead?"
        )
        err_msg3 = (
            "paddle.compat.max() received unexpected keyword argument 'axis'. "
            "\nDid you mean to use paddle.max() instead?"
        )
        err_msg4 = (
            "Non-CUDA GPU placed Tensor does not have 'paddle.float16' op registered.\n"
            "Paddle support following DataTypes: int32, int64, float64, float32, uint8"
        )

        # empty tensor
        empty_tensor = paddle.to_tensor([], dtype='float32')
        with self.assertRaises(ValueError):
            paddle.compat.min(empty_tensor)

        # mixed parameters case 1
        input_ts = paddle.to_tensor([1, 2, 3], dtype='float32')
        other_ts = paddle.to_tensor([1])
        with self.assertRaises(TypeError):
            paddle.compat.min(input_ts, other=other_ts, dim=0)

        # mixed parameters case 2
        with self.assertRaises(TypeError):
            paddle.compat.min(input_ts, 0, other=other_ts)

        # trying to perform grad ops for integral types
        with self.assertRaises(TypeError) as cm:
            tensor = paddle.ones([2, 2], dtype=paddle.int32)
            tensor.stop_gradient = False
            tensors = paddle.compat.max(tensor, dim=0)
        self.assertEqual(str(cm.exception), err_msg1)

        # explicit None case 1
        with self.assertRaises(TypeError) as cm:
            paddle.compat.min(input_ts, dim=None)

        # explicit None case 2
        with self.assertRaises(TypeError) as cm:
            paddle.compat.min(input_ts, None, keepdim=True)

        # keepdim specified without specifying dim
        with self.assertRaises(TypeError) as cm:
            paddle.compat.min(input_ts, keepdim=True)

        # Wrong *args specification case 1
        with self.assertRaises(TypeError) as cm:
            paddle.compat.min(input_ts, False)

        # Wrong *args specification case 2
        with self.assertRaises(TypeError) as cm:
            paddle.compat.min(input_ts, other_ts, True)

        # Tensor input for dim case 1
        with self.assertRaises(TypeError) as cm:
            paddle.compat.min(input_ts, dim=paddle.to_tensor([0]))

        # Tensor input for dim case 2
        with self.assertRaises(TypeError) as cm:
            paddle.compat.min(input_ts, dim=paddle.to_tensor(0))

        # Duplicate Arguments case 1
        with self.assertRaises(TypeError) as cm:
            paddle.compat.max(input_ts, 0, dim=0)

        # Duplicate Arguments case 2
        with self.assertRaises(TypeError) as cm:
            paddle.compat.max(input_ts, other_ts, other=0)

        # Duplicate Arguments case 3
        with self.assertRaises(TypeError) as cm:
            paddle.compat.max(input_ts, dim=0, other=0, keepdim=True)

        # Wrong API used case 1
        with self.assertRaises(TypeError) as cm:
            paddle.min(input=input_ts, dim=0)
        self.assertEqual(str(cm.exception), err_msg2)

        # Wrong API used case 2
        with self.assertRaises(TypeError) as cm:
            paddle.compat.max(input_ts, axis=0)
        self.assertEqual(str(cm.exception), err_msg3)

        # Rejected on CPU types
        with self.assertRaises(TypeError) as cm:
            tensor = paddle.to_tensor([1, 2, 3], dtype="float16")
            cpu_tensor = tensor.to("cpu")
            paddle.compat.max(cpu_tensor, dim=0)
        self.assertEqual(str(cm.exception), err_msg4)


if __name__ == '__main__':
    unittest.main()
