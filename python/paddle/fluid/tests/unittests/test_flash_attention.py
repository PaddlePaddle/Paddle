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

import os
import re
import unittest

import numpy as np

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.nn.functional as F


def get_cuda_version():
    result = os.popen("nvcc --version").read()
    regex = r'release (\S+),'
    match = re.search(regex, result)
    if match:
        num = str(match.group(1))
        integer, decimal = num.split('.')
        return int(integer) * 1000 + int(float(decimal) * 10)
    else:
        return -1


def attention_naive(q, k, v):
    qt = paddle.transpose(q, [0, 2, 1, 3])
    kt = paddle.transpose(k, [0, 2, 1, 3])
    vt = paddle.transpose(v, [0, 2, 1, 3])
    scale = 1.0 / np.sqrt(q.shape[-1])
    s = paddle.matmul(qt, paddle.transpose(kt, [0, 1, 3, 2]))
    s = paddle.scale(s, scale)
    p = F.softmax(s)
    o = paddle.matmul(p, vt)
    return paddle.transpose(o, [0, 2, 1, 3])


@unittest.skipIf(
    not core.is_compiled_with_cuda() or get_cuda_version() < 11030,
    "core is not compiled with CUDA and cuda version need larger than or equal to 11.3",
)
class TestFlashAttentionAPI(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        self.shape = (2, 128, 8, 16)
        self.blocksize = 2
        self.dtype = 'float16'
        self.dropout = 0.0
        self.causal = False
        self.return_softmax = False

    def test_all(self):
        paddle.disable_static()

        query = np.random.random(self.shape)
        key = np.random.random(self.shape)
        value = np.random.random(self.shape)

        q = paddle.to_tensor(query, place=self.place, dtype=self.dtype)
        k = paddle.to_tensor(key, place=self.place, dtype=self.dtype)
        v = paddle.to_tensor(value, place=self.place, dtype=self.dtype)

        out, _ = F.flash_attention(
            q, k, v, self.dropout, self.causal, self.return_softmax
        )
        numpy_result = attention_naive(q, k, v)

        np.testing.assert_allclose(
            out.numpy(), numpy_result, rtol=5e-03, atol=1e-03
        )

        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            qs = paddle.static.data(
                name="q", shape=self.shape, dtype=self.dtype
            )
            ks = paddle.static.data(
                name="k", shape=self.shape, dtype=self.dtype
            )
            vs = paddle.static.data(
                name="v", shape=self.shape, dtype=self.dtype
            )

            outs, softmax = F.flash_attention(
                qs, ks, vs, self.dropout, self.causal, self.return_softmax
            )

            exe = fluid.Executor(self.place)
            fetches_result = exe.run(
                feed={
                    "q": query.astype('float16'),
                    "k": key.astype('float16'),
                    "v": value.astype('float16'),
                },
                fetch_list=[outs],
            )

            np.testing.assert_allclose(
                fetches_result[0], numpy_result, rtol=5e-03, atol=1e-03
            )


class TestFlashAttentionAPITest128(TestFlashAttentionAPI):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        self.shape = (2, 128, 8, 16)
        self.blocksize = 2
        self.dtype = paddle.float16
        self.dropout = 0.0
        self.causal = False
        self.return_softmax = False


class TestFlashAttentionAPITest256(TestFlashAttentionAPI):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        self.shape = (2, 256, 8, 16)
        self.blocksize = 2
        self.dtype = paddle.float16
        self.dropout = 0.0
        self.causal = False
        self.return_softmax = False


class TestFlashAttentionAPITest512(TestFlashAttentionAPI):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        self.shape = (2, 512, 8, 16)
        self.blocksize = 2
        self.dtype = paddle.float16
        self.dropout = 0.0
        self.causal = False
        self.return_softmax = False


class TestFlashAttentionBackward(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        self.shape = (2, 512, 8, 16)
        self.blocksize = 2
        self.dtype = paddle.float16
        self.dropout = 0.0
        self.causal = False
        self.return_softmax = False

    def test_backward(self):
        paddle.disable_static()

        query = np.random.random(self.shape)
        key = np.random.random(self.shape)
        value = np.random.random(self.shape)

        q = paddle.to_tensor(query, place=self.place, dtype=self.dtype)
        k = paddle.to_tensor(key, place=self.place, dtype=self.dtype)
        v = paddle.to_tensor(value, place=self.place, dtype=self.dtype)
        q.stop_gradient = False
        k.stop_gradient = False
        v.stop_gradient = False

        out, _ = F.flash_attention(
            q, k, v, self.dropout, self.causal, self.return_softmax
        )
        out.backward()
        self.assertEqual(q.grad.shape, q.shape)


if __name__ == '__main__':
    unittest.main()
