# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import _C_ops


def compute_index_put_ref(x_np, indices_np, value_np):
    x_np[indices_np] = value_np
    return x_np


def raw_index_put(x, indices, value):
    return _C_ops.index_put(x, indices, value)


def gen_indices_np(x_shape, indices_shapes, index_type):
    indices = []
    if index_type == np.bool_:
        indices = np.zeros(indices_shapes[0], dtype=np.bool_)
        indices.flatten()
        for i in range(len(indices)):
            indices[i] = (i & 1) == 0
        indices = indices.reshape(indices_shapes[0])

    else:
        for i in range(len(indices_shapes)):
            index_np = np.random.randint(
                low=0, high=x_shape[i], size=indices_shapes[i], dtype=index_type
            )
            indices.append(index_np)
    return indices


class TestIndexPutOp(unittest.TestCase):
    def setUp(self):
        self.init_dtype_type()
        self.x_np = np.random.random(self.x_shape).astype(self.dtype_np)
        self.value_np = np.random.random(self.value_shape).astype(self.dtype_np)
        self.indices_np = gen_indices_np(
            self.x_shape, self.indices_shapes, self.index_type_np
        )

        self.x_pd = paddle.to_tensor(self.x_np, dtype=self.dtype_pd)
        self.value_pd = paddle.to_tensor(self.value_np, dtype=self.dtype_pd)
        self.indices_pd = [
            paddle.to_tensor(indice, dtype=self.index_type_pd)
            for indice in self.indices_np
        ]

    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.int64
        self.x_shape = (100, 110)
        self.indices_shapes = ((21,), (21,))
        self.value_shape = (21,)
        self.dtype_pd = paddle.float64
        self.index_type_pd = paddle.int64
        self.dtype_pd = paddle.float64

    def test_forward(self):
        ref_res = compute_index_put_ref(
            self.x_np, self.indices_np, self.value_np
        )
        pd_res = raw_index_put(self.x_pd, self.indices_pd, self.value_pd)
        np.testing.assert_allclose(ref_res, pd_res.numpy(), atol=1e-7)

    def test_backward(self):
        value = paddle.ones(shape=[4], dtype=self.dtype_pd)
        x = paddle.ones(shape=[16, 21], dtype=self.dtype_pd)
        ix1 = paddle.to_tensor([0, 1, 2, 3], dtype=self.index_type_pd)
        ix2 = paddle.to_tensor([0, 1, 2, 3], dtype=self.index_type_pd)
        value.stop_gradient = False
        x[ix1, ix2] = value

        dvalue = paddle.grad(
            outputs=[x], inputs=[value], create_graph=False, retain_graph=True
        )[0]

        np.testing.assert_allclose(
            np.array([1.0, 1.0, 1.0, 1.0], dtype=self.dtype_np),
            dvalue.numpy(),
            atol=1e-7,
        )

    def test_backward1(self):
        value = paddle.ones(shape=[1], dtype=self.dtype_pd)
        x = paddle.ones(shape=[16, 21], dtype=self.dtype_pd)
        ix1 = paddle.to_tensor([0, 1, 2, 3], dtype=self.index_type_pd)
        ix2 = paddle.to_tensor([0, 1, 2, 3], dtype=self.index_type_pd)
        value.stop_gradient = False
        x[ix1, ix2] = value

        dvalue = paddle.grad(
            outputs=[x], inputs=[value], create_graph=False, retain_graph=True
        )[0]

        np.testing.assert_allclose(
            np.array([4.0], dtype=self.dtype_np), dvalue.numpy(), atol=1e-7
        )


class TestIndexPutOpFloat32(TestIndexPutOp):
    def init_dtype_type(self):
        self.dtype_np = np.float32
        self.index_type_np = np.int64
        self.x_shape = (100, 110)
        self.indices_shapes = ((21,), (21,))
        self.value_shape = (21,)
        self.dtype_pd = paddle.float32
        self.index_type_pd = paddle.int64
        self.dtype_pd = paddle.float32


class TestIndexPutOpFloat16(TestIndexPutOp):
    def init_dtype_type(self):
        self.dtype_np = np.float16
        self.index_type_np = np.int64
        self.x_shape = (100, 110)
        self.indices_shapes = ((21,), (21,))
        self.value_shape = (21,)
        self.dtype_pd = paddle.float16
        self.index_type_pd = paddle.int64
        self.dtype_pd = paddle.float16


class TestIndexPutOpInt32(TestIndexPutOp):
    def init_dtype_type(self):
        self.dtype_np = np.int32
        self.index_type_np = np.int64
        self.x_shape = (100, 110)
        self.indices_shapes = ((21,), (21,))
        self.value_shape = (21,)
        self.dtype_pd = paddle.int32
        self.index_type_pd = paddle.int64
        self.dtype_pd = paddle.int32


class TestIndexPutOpInt64(TestIndexPutOp):
    def init_dtype_type(self):
        self.dtype_np = np.int64
        self.index_type_np = np.int64
        self.x_shape = (100, 110)
        self.indices_shapes = ((21,), (21,))
        self.value_shape = (21,)
        self.dtype_pd = paddle.int64
        self.index_type_pd = paddle.int64
        self.dtype_pd = paddle.int64


class TestIndexPutOpBool(TestIndexPutOp):
    def init_dtype_type(self):
        self.dtype_np = np.bool_
        self.index_type_np = np.int64
        self.x_shape = (100, 110)
        self.indices_shapes = ((21,), (21,))
        self.value_shape = (21,)
        self.dtype_pd = paddle.bool
        self.index_type_pd = paddle.int64
        self.dtype_pd = paddle.bool


if __name__ == '__main__':
    unittest.main()
