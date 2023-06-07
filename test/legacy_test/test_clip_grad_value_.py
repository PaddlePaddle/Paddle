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
from paddle.nn.utils.clip_grad_value_ import clip_grad_value_


class TestClipGradValue(unittest.TestCase):
    def test_basic(self):
        run_test_equal(
            self,
            shape=[16, 16],
            dtype=np.float32,
            clip_value=1,
        )
        run_test_equal(
            self,
            shape=(100,),
            dtype=np.float32,
            clip_value=3.5,
        )
        run_test_equal(self, shape=[4, 8, 16], dtype=np.float32, clip_value=0)

    def test_errors(self):
        def TestValueError():
            input_pd = paddle.to_tensor(
                np.random.random([1, 2]).astype(np.float32)
            )
            input_pd.grad = paddle.to_tensor(
                np.random.random([1, 2]).astype(np.float32)
            )
            clip_grad_value_(input_pd, clip_value=-1)

        self.assertRaises(ValueError, TestValueError)

        def TestRuntimeErrorStaticMode():
            paddle.enable_static()
            input_pd = paddle.to_tensor(
                np.random.random([1, 2]).astype(np.float32)
            )
            input_pd.grad = paddle.to_tensor(
                np.random.random([1, 2]).astype(np.float32)
            )
            clip_grad_value_(input_pd, clip_value=1)
            paddle.disable_static()

        self.assertRaises(RuntimeError, TestRuntimeErrorStaticMode)


def run_test_equal(
    self,
    shape,
    dtype,
    clip_value,
):
    input = np.random.random(shape).astype(dtype)
    grad = np.random.random(shape).astype(dtype)
    input_pd = paddle.to_tensor(input)
    input_pd.grad = paddle.to_tensor(grad)

    output = np.clip(grad, a_min=-clip_value, a_max=clip_value)
    clip_grad_norm_result = clip_grad_value_(
        input_pd,
        clip_value=clip_value,
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
