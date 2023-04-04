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

import copy
import unittest

import numpy as np

import paddle
from paddle import _C_ops


def compute_index_put_ref(x_np, indices_np, value_np, accumulate=False):
    if accumulate:
        x_np[indices_np] += value_np
        return x_np
    else:
        x_np[indices_np] = value_np
        return x_np


def raw_index_put(x, indices, value):
    return _C_ops.index_put(x, indices, value)


def has_duplicate_index(indices, shapes):
    bd_shape = np.broadcast_shapes(*shapes)
    bd_indices = [
        list(np.broadcast_to(indice, bd_shape).flatten()) for indice in indices
    ]

    zip_res = list(zip(*bd_indices))
    if len(zip_res) == len(set(zip_res)):
        return False
    else:
        return True


def gen_indices_np(x_shape, indices_shapes, index_type):
    indices = []
    if index_type == np.bool_:
        indice = np.zeros(indices_shapes[0], dtype=np.bool_)
        indice.flatten()
        for i in range(len(indice)):
            indice[i] = (i & 1) == 0
        indice = indice.reshape(indices_shapes[0])
        indices.append(indice)
    else:
        while True:
            indices = []
            for i in range(len(indices_shapes)):
                np.random.seed()
                index_np = np.random.randint(
                    low=0,
                    high=x_shape[i],
                    size=indices_shapes[i],
                    dtype=index_type,
                )
                indices.append(index_np)
            if not has_duplicate_index(
                copy.deepcopy(indices), copy.deepcopy(indices_shapes)
            ):
                break
    return tuple(indices)


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
        self.indices_shapes = [(21,), (21,)]
        self.value_shape = (21,)
        self.dtype_pd = paddle.float64
        self.index_type_pd = paddle.int64

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

    def test_backwardScalarVal(self):
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
        self.indices_shapes = [(21,), (21,)]
        self.value_shape = (21,)
        self.dtype_pd = paddle.float32
        self.index_type_pd = paddle.int64
        self.dtype_pd = paddle.float32


class TestIndexPutOpFloat16(TestIndexPutOp):
    def init_dtype_type(self):
        self.dtype_np = np.float16
        self.index_type_np = np.int64
        self.x_shape = (100, 110)
        self.indices_shapes = [(21,), (21,)]
        self.value_shape = (21,)
        self.dtype_pd = paddle.float16
        self.index_type_pd = paddle.int64
        self.dtype_pd = paddle.float16


class TestIndexPutOpInt32(TestIndexPutOp):
    def init_dtype_type(self):
        self.dtype_np = np.int32
        self.index_type_np = np.int64
        self.x_shape = (100, 110)
        self.indices_shapes = [(21,), (21,)]
        self.value_shape = (21,)
        self.dtype_pd = paddle.int32
        self.index_type_pd = paddle.int64
        self.dtype_pd = paddle.int32


class TestIndexPutOpInt64(TestIndexPutOp):
    def init_dtype_type(self):
        self.dtype_np = np.int64
        self.index_type_np = np.int64
        self.x_shape = (100, 110)
        self.indices_shapes = [(21,), (21,)]
        self.value_shape = (21,)
        self.dtype_pd = paddle.int64
        self.index_type_pd = paddle.int64
        self.dtype_pd = paddle.int64


class TestIndexPutOpBool(TestIndexPutOp):
    def init_dtype_type(self):
        self.dtype_np = np.bool_
        self.index_type_np = np.int64
        self.x_shape = (100, 110)
        self.indices_shapes = [(21,), (21,)]
        self.value_shape = (21,)
        self.dtype_pd = paddle.bool
        self.index_type_pd = paddle.int64
        self.dtype_pd = paddle.bool


class TestIndexPutAPIBase(unittest.TestCase):
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
        self.indices_pd = tuple(self.indices_pd)

    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.int64
        self.x_shape = (100, 110)
        self.indices_shapes = [(21,), (21,)]
        self.value_shape = (21,)
        self.dtype_pd = paddle.float64
        self.index_type_pd = paddle.int64
        self.accumulate = False


class TestIndexPutAPI0(TestIndexPutAPIBase):
    def test_forward(self):
        ref_res = compute_index_put_ref(
            self.x_np, self.indices_np, self.value_np
        )
        pd_res = paddle.index_put(
            self.x_pd, self.indices_pd, self.value_pd, self.accumulate
        )
        np.testing.assert_allclose(ref_res, pd_res.numpy(), atol=1e-7)


class TestIndexPutAPI1(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.int64
        self.x_shape = (110, 42, 56, 56)
        self.indices_shapes = ((16, 16), (16, 16), (1, 16), (1, 16))
        self.value_shape = (16, 16)
        self.dtype_pd = paddle.float64
        self.index_type_pd = paddle.int64
        self.accumulate = False

    def test_forward(self):
        ref_res = compute_index_put_ref(
            self.x_np, self.indices_np, self.value_np
        )
        pd_res = paddle.index_put(
            self.x_pd, self.indices_pd, self.value_pd, self.accumulate
        )
        np.testing.assert_allclose(ref_res, pd_res.numpy(), atol=1e-7)


class TestIndexPutAPI2(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.bool_
        self.x_shape = (110, 94)
        self.indices_shapes = [(110, 94)]
        self.value_shape = 5170
        self.dtype_pd = paddle.float64
        self.index_type_pd = paddle.bool
        self.accumulate = False

    def test_forward(self):
        ref_res = compute_index_put_ref(
            self.x_np, self.indices_np, self.value_np
        )
        pd_res = paddle.index_put(
            self.x_pd, self.indices_pd, self.value_pd, self.accumulate
        )
        np.testing.assert_allclose(ref_res, pd_res.numpy(), atol=1e-7)


class TestIndexPutAPIBackward0(TestIndexPutAPIBase):
    def test_backward(self):
        value = paddle.ones(shape=[4], dtype=self.dtype_pd)
        x = paddle.ones(shape=[16, 21], dtype=self.dtype_pd)
        ix1 = paddle.to_tensor([0, 1, 2, 3], dtype=self.index_type_pd)
        ix2 = paddle.to_tensor([0, 1, 2, 3], dtype=self.index_type_pd)
        value.stop_gradient = False
        x.stop_gradient = False
        out = paddle.index_put(x, (ix1, ix2), value, self.accumulate)

        dx, dvalue = paddle.grad(
            outputs=[out],
            inputs=[x, value],
            create_graph=False,
            retain_graph=True,
        )
        ref_dx = np.ones(shape=[16, 21], dtype=self.dtype_np)
        ref_dx[ix1, ix2] = 0

        np.testing.assert_allclose(ref_dx, dx.numpy(), atol=1e-7)

        np.testing.assert_allclose(ref_dx, dx.numpy(), atol=1e-7)
        np.testing.assert_allclose(
            np.array([1.0, 1.0, 1.0, 1.0], dtype=self.dtype_np),
            dvalue.numpy(),
            atol=1e-7,
        )


class TestIndexPutAPIBackward1(TestIndexPutAPIBase):
    def test_backwardScalarVal(self):
        value = paddle.ones(shape=[1], dtype=self.dtype_pd)
        x = paddle.ones(shape=[16, 21], dtype=self.dtype_pd)
        ix1 = paddle.to_tensor([0, 1, 2, 3], dtype=self.index_type_pd)
        ix2 = paddle.to_tensor([0, 1, 2, 3], dtype=self.index_type_pd)
        value.stop_gradient = False
        x.stop_gradient = False
        out = paddle.index_put(x, (ix1, ix2), value, self.accumulate)

        dx, dvalue = paddle.grad(
            outputs=[out],
            inputs=[x, value],
            create_graph=False,
            retain_graph=True,
        )
        ref_dx = np.ones(shape=[16, 21], dtype=self.dtype_np)
        ref_dx[ix1, ix2] = 0

        np.testing.assert_allclose(ref_dx, dx.numpy(), atol=1e-7)
        np.testing.assert_allclose(
            np.array([4.0], dtype=self.dtype_np), dvalue.numpy(), atol=1e-7
        )


if __name__ == '__main__':
    unittest.main()
