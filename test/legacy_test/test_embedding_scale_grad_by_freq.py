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
        self.places = [paddle.CPUPlace()]
        if paddle.core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

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


if __name__ == '__main__':
    unittest.main()
