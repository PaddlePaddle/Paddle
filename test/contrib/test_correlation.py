# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import base

paddle.enable_static()


def corr(
    x_1,
    x_2,
    pad_size=4,
    kernel_size=1,
    max_displacement=4,
    stride1=1,
    stride2=1,
    corr_multiply=1,
):
    K = kernel_size

    rinput1 = np.pad(
        x_1,
        ((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size)),
        mode='constant',
    )
    rinput2 = np.pad(
        x_2,
        ((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size)),
        mode='constant',
    )
    rinput1 = np.transpose(rinput1, (0, 2, 3, 1))
    rinput2 = np.transpose(rinput2, (0, 2, 3, 1))
    B = int(rinput1.shape[0])
    H = int(x_1.shape[2])
    W = int(x_2.shape[3])
    d = max_displacement
    D = 2 * d + 1
    output = np.zeros((B, D * D, H, W), dtype=np.float32)

    for b in range(B):
        for i in range(H):
            for j in range(W):
                for k in range(-d, d + 1):
                    for l in range(-d, d + 1):
                        x1_index = i + pad_size
                        y1_index = j + pad_size
                        x2_index = x1_index + k
                        y2_index = y1_index + l
                        output[b, l + d + D * (k + d), i, j] = np.mean(
                            rinput1[
                                b,
                                x1_index : x1_index + K,
                                y1_index : y1_index + K,
                            ]
                            * rinput2[
                                b,
                                x2_index : x2_index + K,
                                y2_index : y2_index + K,
                            ]
                        )

    return output


class TestCorrelationOp(unittest.TestCase):
    def test_check_output(self):
        if not base.core.is_compiled_with_cuda():
            return
        np.random.seed(13)
        np.set_printoptions(threshold=np.inf)
        x_shape = (2, 10, 3, 3)
        x_type = 'float32'
        x1 = paddle.static.data(
            name='x1',
            shape=x_shape,
            dtype=x_type,
        )
        x1.desc.set_need_check_feed(False)
        x1.stop_gradient = False
        x2 = paddle.static.data(
            name='x2',
            shape=x_shape,
            dtype=x_type,
        )
        x2.desc.set_need_check_feed(False)
        x2.stop_gradient = False

        x1_np = np.random.randn(2, 3, 4, 5).astype(x_type)
        x2_np = np.random.randn(2, 3, 4, 5).astype(x_type)
        out_np = corr(
            x1_np,
            x2_np,
            pad_size=4,
            kernel_size=1,
            max_displacement=4,
            stride1=1,
            stride2=1,
        )

        out = paddle.incubate.layers.correlation(
            x1,
            x2,
            pad_size=4,
            kernel_size=1,
            max_displacement=4,
            stride1=1,
            stride2=1,
        )

        loss = paddle.mean(out)
        optimizer = paddle.optimizer.Momentum(0.0001, 0.9)
        optimizer.minimize(loss)

        place = base.CUDAPlace(0)
        exe = base.Executor(place)
        res = exe.run(feed={'x1': x1_np, 'x2': x2_np}, fetch_list=[out, loss])

        np.testing.assert_allclose(res[0], out_np, rtol=1e-05, atol=1e-8)


class Net(paddle.nn.Layer):
    def __init__(self, name_scope):
        super().__init__(name_scope)

    def forward(self, x1, x2):
        y = paddle.incubate.layers.correlation(
            x1,
            x2,
            pad_size=4,
            kernel_size=1,
            max_displacement=4,
            stride1=1,
            stride2=1,
        )
        return y


class TestCorrelationOpDyGraph(unittest.TestCase):
    def test_check_output(self):
        if not base.core.is_compiled_with_cuda():
            return
        np.random.seed(13)
        np.set_printoptions(threshold=np.inf)
        x_shape = (2, 10, 3, 3)
        x_type = 'float32'
        place = base.CUDAPlace(0)
        with base.dygraph.guard(place):
            x1_np = np.random.randn(2, 3, 4, 5).astype(x_type)
            x2_np = np.random.randn(2, 3, 4, 5).astype(x_type)
            out_np = corr(
                x1_np,
                x2_np,
                pad_size=4,
                kernel_size=1,
                max_displacement=4,
                stride1=1,
                stride2=1,
            )

            x1 = paddle.to_tensor(x1_np)
            x2 = paddle.to_tensor(x2_np)
            corr_pd = Net('corr_pd')
            y = corr_pd(x1, x2)
            out = y.numpy()
            np.testing.assert_allclose(out, out_np, rtol=1e-05, atol=1e-8)


if __name__ == '__main__':
    unittest.main()
