# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn.functional as F

np.random.seed(10)


def stable_softmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    # clip to shiftx, otherwise, when calc loss with
    # log(exp(shiftx)), may get log(0)=INF
    shiftx = (x - np.max(x)).clip(-64.0)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


def ref_softmax(x, axis=None, dtype=None):
    x_t = x.copy()
    if dtype is not None:
        x_t = x_t.astype(dtype)
    if axis is None:
        axis = -1
    return np.apply_along_axis(stable_softmax, axis, x_t)


class TestSoftmax(unittest.TestCase):
    def setUp(self):
        self.shape = [2, 4, 6]
        self.dtype = 'float32'
        self.f_x = np.random.uniform(-1.0, 1.0, self.shape).astype(self.dtype)

    def test_softmax_shape(self):
        f_x1 = np.random.uniform(-1.0, 1.0, [2, 3, 4, 5]).astype('float32')

        p_x1 = paddle.to_tensor(f_x1)
        p_out1 = F.softmax(p_x1)

        # f_out1 = ref_softmax(f_x1, axis=-1, dtype=None)
        # np.testing.assert_allclose(p_out1.numpy(), f_out1, rtol=1e-05)

        # for 0-sized Tensor
        # f_x2 = np.random.uniform(-1.0, 1.0, [0]).astype('float32')

        # p_x2 = paddle.to_tensor(f_x2)
        # p_out2 = F.softmax(p_x2)
        # print(p_out2)

        # f_out2 = ref_softmax([], axis=-1, dtype=None)
        # np.testing.assert_allclose(p_out2.numpy(), f_out2, rtol=1e-05)

    # def test_softmax_dtype(self):
    #     f_x_fp16 = np.random.uniform(-1.0, 1.0, [2, 3, 4, 5]).astype('float16')

    #     p_x_fp16 = paddle.to_tensor(f_x_fp16)
    #     p_out1 = F.softmax(p_x_fp16)

    #     f_out1 = ref_softmax(f_x_fp16, axis=-1, dtype=None)
    #     np.testing.assert_allclose(p_out1.numpy(), f_out1, rtol=1e-05)


if __name__ == "__main__":
    unittest.main()
