#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import get_places

import paddle
from paddle.nn.functional import embedding


def ref_embedding_scale_grad_(x, weight_unscaled_grad):
    grad = np.zeros_like(weight_unscaled_grad)
    unique, count = np.unique(x, return_counts=True)
    count_dict = dict(zip(unique, count))
    for k, v in count_dict.items():
        grad[k] = weight_unscaled_grad[k] / v
    return grad


class TestEmbeddingAPIScaleGradByFreq(unittest.TestCase):
    def setUp(self):
        self.init_data()
        self.places = get_places()

    def init_data(self):
        self.dtype = "float32"
        self.x_np = np.array([[2, 1, 3], [4, 5, 6]]).astype("int64")
        self.weight_np = np.random.random((10, 4)).astype(self.dtype)
        self.padding_idx = -1

    def test_scale_grad_dygraph(self):
        for place in self.places:
            paddle.disable_static(place)
            x = paddle.to_tensor(self.x_np)
            w1 = paddle.to_tensor(self.weight_np)
            w1.stop_gradient = False
            w2 = paddle.to_tensor(np.copy(self.weight_np))
            w2.stop_gradient = False
            unscale_out = embedding(
                x, w1, padding_idx=self.padding_idx, scale_grad_by_freq=False
            )
            unscale_out.backward()
            unscale_grad = w1.grad.numpy()
            scale_out = embedding(
                x, w2, padding_idx=self.padding_idx, scale_grad_by_freq=True
            )
            scale_out.backward()
            scale_grad = w2.grad.numpy()
            scale_grad_ref = ref_embedding_scale_grad_(self.x_np, unscale_grad)
            np.testing.assert_allclose(scale_grad_ref, scale_grad)
            np.testing.assert_equal(scale_out.numpy(), unscale_out.numpy())
            paddle.enable_static()

    def test_scale_grad_static(self):
        paddle.enable_static()
        for place in self.places:
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x = paddle.static.data("x", self.x_np.shape, self.x_np.dtype)
                w1 = paddle.static.data("w1", self.weight_np.shape, self.dtype)
                w1.stop_gradient = False
                w2 = paddle.static.data("w2", self.weight_np.shape, self.dtype)
                w2.stop_gradient = False
                out1 = embedding(
                    x,
                    w1,
                    padding_idx=self.padding_idx,
                    scale_grad_by_freq=False,
                )
                w1_grad = paddle.static.gradients([out1], w1)
                out2 = embedding(
                    x, w2, padding_idx=self.padding_idx, scale_grad_by_freq=True
                )
                w2_grad = paddle.static.gradients([out2], w2)
                exe = paddle.static.Executor(place)
                [unscale_grad, scale_grad, unscale_out, scale_out] = exe.run(
                    feed={
                        "x": self.x_np,
                        "w1": self.weight_np,
                        "w2": np.copy(self.weight_np),
                    },
                    fetch_list=[w1_grad, w2_grad, out1, out2],
                    return_numpy=True,
                )
            scale_grad_ref = ref_embedding_scale_grad_(self.x_np, unscale_grad)
            np.testing.assert_allclose(scale_grad_ref, scale_grad)
            np.testing.assert_allclose(unscale_out, scale_out)


class TestEmbeddingAPIScaleGradByFreq1(TestEmbeddingAPIScaleGradByFreq):
    def init_data(self):
        self.dtype = "float32"
        self.x_np = np.array([[2, 1, 2, 3], [1, 5, 6, 1]]).astype("int64")
        self.weight_np = np.random.random((10, 4)).astype(self.dtype)
        self.padding_idx = 2


class TestEmbeddingAPIScaleGradByFreq2(TestEmbeddingAPIScaleGradByFreq):
    def init_data(self):
        self.dtype = "float32"
        self.x_np = np.array(
            [[2, 1, 3], [2, 1, 3], [2, 1, 3], [2, 1, 3], [4, 5, 6], [4, 5, 6]]
        ).astype("int32")
        self.weight_np = np.random.random((10, 4)).astype(self.dtype)
        self.padding_idx = 5


class TestEmbeddingAPIScaleGradByFreqError(unittest.TestCase):
    def test_argument_error(self):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.static.data("x", [2, 4], "float32")
            w = paddle.static.data("w", [10, 4], "int32")
            self.assertRaises(
                AttributeError,
                embedding,
                x,
                w,
                sparse=True,
                scale_grad_by_freq=True,
            )


class TestEmbeddingScaleGradByFreqZeroSize(unittest.TestCase):
    """Test scale_grad_by_freq=True with 0-size input tensors.

    When input has 0 elements (e.g. shape [0, 3]), the backward kernel
    used to launch CountFreqKernel with GET_BLOCKS(0)=0 grid blocks,
    causing CUDA error(9). This test ensures the fix works correctly.
    """

    def setUp(self):
        self.places = get_places()
        self.weight_np = np.random.random((10, 4)).astype("float32")

    def _check_zero_size_dygraph(self, x_shape, x_dtype, padding_idx):
        for place in self.places:
            paddle.disable_static(place)
            x = paddle.zeros(x_shape, dtype=x_dtype)
            w = paddle.to_tensor(self.weight_np)
            w.stop_gradient = False

            out = embedding(
                x, w, padding_idx=padding_idx, scale_grad_by_freq=True
            )
            # Output shape: x_shape + [embed_dim]
            expected_out_shape = [*x_shape, self.weight_np.shape[1]]
            self.assertEqual(list(out.shape), expected_out_shape)

            out.backward()
            # Weight grad must have the same shape as weight and be all-zeros
            self.assertIsNotNone(w.grad)
            self.assertEqual(list(w.grad.shape), list(self.weight_np.shape))
            np.testing.assert_array_equal(
                w.grad.numpy(), np.zeros_like(self.weight_np)
            )
            paddle.enable_static()

    def test_zero_first_dim_int32(self):
        # shape [0, 3] int32, padding_idx=5
        self._check_zero_size_dygraph([0, 3], 'int32', 5)

    def test_zero_first_dim_int64(self):
        # shape [0, 3] int64, padding_idx=-1
        self._check_zero_size_dygraph([0, 3], 'int64', -1)

    def test_zero_first_dim_only(self):
        # shape [0] int64
        self._check_zero_size_dygraph([0], 'int64', 2)

    def test_zero_second_dim_int64(self):
        # shape [2, 0] int64, padding_idx=2
        self._check_zero_size_dygraph([2, 0], 'int64', 2)

    def test_zero_second_dim_int32(self):
        # shape [6, 0] int32, padding_idx=5
        self._check_zero_size_dygraph([6, 0], 'int32', 5)

    def test_zero_size_static(self):
        """Verify 0-size input works in static graph mode too."""
        paddle.enable_static()
        x_shape = [0, 3]
        x_dtype = 'int64'
        for place in self.places:
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x = paddle.static.data("x", x_shape, x_dtype)
                w = paddle.static.data(
                    "w", self.weight_np.shape, self.weight_np.dtype
                )
                w.stop_gradient = False
                out = embedding(x, w, padding_idx=-1, scale_grad_by_freq=True)
                w_grad = paddle.static.gradients([out], w)
                exe = paddle.static.Executor(place)
                x_val = np.zeros(x_shape, dtype=x_dtype)
                [out_val, grad_val] = exe.run(
                    feed={"x": x_val, "w": self.weight_np},
                    fetch_list=[out, w_grad],
                    return_numpy=True,
                )
            expected_out_shape = [*x_shape, self.weight_np.shape[1]]
            self.assertEqual(list(out_val.shape), expected_out_shape)
            np.testing.assert_array_equal(
                grad_val, np.zeros_like(self.weight_np)
            )


if __name__ == '__main__':
    unittest.main()
