#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.nn.utils.clip_grad_norm_ import clip_grad_norm_


class TestClipGradNorm(unittest.TestCase):
    def test_basic(self):
        run_test_equal(
            self,
            shape=[16, 16],
            dtype=np.float32,
            max_norm=5,
            norm_type=2,
        )
        run_test_equal(
            self,
            shape=(100,),
            dtype=np.float32,
            max_norm=1e20,
            norm_type=2,
        )
        run_test_equal(
            self,
            shape=[4, 8, 16],
            dtype=np.float32,
            max_norm=1.0,
            norm_type=float("inf"),
        )

    def test_errors(self):
        def TestValueError():
            input_pd = paddle.to_tensor(
                np.random.random([1, 2]).astype(np.float32)
            )
            input_pd.grad = paddle.to_tensor(
                np.random.random([1, 2]).astype(np.float32)
            )
            clip_grad_norm_(input_pd, max_norm=2, norm_type=float("-inf"))

        self.assertRaises(ValueError, TestValueError)

        def TestRuntimeError():
            input_pd = paddle.to_tensor(
                np.random.random([1, 2]).astype(np.float32)
            )
            input_pd.grad = paddle.full([1, 2], float("inf"))
            clip_grad_norm_(
                input_pd, max_norm=2, norm_type=2, error_if_nonfinite=True
            )

        self.assertRaises(RuntimeError, TestRuntimeError)

        def TestRuntimeErrorStaticMode():
            paddle.enable_static()
            input_pd = paddle.to_tensor(
                np.random.random([1, 2]).astype(np.float32)
            )
            input_pd.grad = paddle.to_tensor(
                np.random.random([1, 2]).astype(np.float32)
            )
            clip_grad_norm_(input_pd, max_norm=2, norm_type=float("inf"))
            paddle.disable_static()

        if paddle.framework.use_pir_api():
            self.assertRaises(AttributeError, TestRuntimeErrorStaticMode)
        else:
            self.assertRaises(RuntimeError, TestRuntimeErrorStaticMode)


def run_test_equal(
    self,
    shape,
    dtype,
    max_norm,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
):
    input = np.random.random(shape).astype(dtype)
    grad = np.random.random(shape).astype(dtype)
    input_pd = paddle.to_tensor(input)
    input_pd.grad = paddle.to_tensor(grad)

    if norm_type == 2:
        grad = grad.reshape(1, grad.size)
        output = np.linalg.norm(grad, 'fro')
    elif norm_type == np.inf:
        output = np.amax(np.abs(grad))
    else:
        output = np.linalg.norm(grad, norm_type)
    clip_grad_norm_result = clip_grad_norm_(
        input_pd,
        max_norm=max_norm,
        norm_type=norm_type,
        error_if_nonfinite=error_if_nonfinite,
    )

    np.testing.assert_allclose(
        clip_grad_norm_result.numpy(),
        output,
        rtol=1e-05,
        atol=1e-05,
        equal_nan=False,
    )


if __name__ == '__main__':
    unittest.main()
