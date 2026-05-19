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

import copy
import unittest

import numpy as np
from op_test import get_device_place, get_devices, is_custom_device

import paddle
from paddle.base import core


def compute_index_get_ref(x_np, indices_np):
    return x_np[tuple(indices_np)]


def compute_index_get_grad_ref(x_np, indices_np, out_grad_np):
    x_grad = np.zeros_like(x_np)
    np.add.at(x_grad, tuple(indices_np), out_grad_np)
    return x_grad


def raw_index_get(x, indices):
    return paddle.index_get(x, indices)


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


def gen_indices_np(x_shape, indices_shapes, index_type, is_all_false):
    indices = []
    if index_type == np.bool_:
        indice = np.zeros(indices_shapes[0], dtype=np.bool_)
        if not is_all_false:
            indice = indice.flatten()
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


class TestIndexGetAPIBase(unittest.TestCase):
    def setUp(self):
        self.mixed_indices = False
        self.is_all_false = False
        self.init_dtype_type()
        self.setPlace()
        self.x_np = np.random.random(self.x_shape).astype(self.dtype_np)

        if self.mixed_indices:
            tmp_indices_np1 = gen_indices_np(
                self.x_shape,
                self.indices_shapes,
                self.index_type_np,
                self.is_all_false,
            )
            tmp_indices_np2 = gen_indices_np(
                self.x_shape,
                self.indices_shapes1,
                self.index_type_np1,
                self.is_all_false,
            )
            self.indices_np = tuple(
                list(tmp_indices_np1) + list(tmp_indices_np2)
            )
        else:
            self.indices_np = gen_indices_np(
                self.x_shape,
                self.indices_shapes,
                self.index_type_np,
                self.is_all_false,
            )

    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.int64
        self.x_shape = (100, 110)
        self.indices_shapes = [(21,), (21,)]
        self.dtype_pd = "float64"
        self.index_type_pd = "int64"

    def setPlace(self):
        self.place = get_devices()
        if self.dtype_np is np.float16 and "cpu" in self.place:
            self.place.remove("cpu")

    def test_dygraph_forward(self):
        paddle.disable_static()
        for place in self.place:
            paddle.device.set_device(place)
            self.x_pd = paddle.to_tensor(self.x_np, dtype=self.dtype_pd)
            self.indices_pd = [
                paddle.to_tensor(indice) for indice in self.indices_np
            ]
            self.indices_pd = tuple(self.indices_pd)
            ref_res = compute_index_get_ref(self.x_np, self.indices_np)
            pd_res = paddle.index_get(self.x_pd, self.indices_pd)
            np.testing.assert_allclose(ref_res, pd_res.numpy(), atol=1e-7)

    def test_static_forward(self):
        paddle.enable_static()
        for place in self.place:
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data(
                    name="x", shape=self.x_shape, dtype=self.dtype_pd
                )
                if self.mixed_indices:
                    indices = tuple(
                        [
                            paddle.static.data(
                                name="indice" + str(i),
                                shape=self.indices_shapes[i],
                                dtype=self.index_type_pd,
                            )
                            for i in range(len(self.indices_shapes))
                        ]
                        + [
                            paddle.static.data(
                                name="indice"
                                + str(i + len(self.indices_shapes)),
                                shape=self.indices_shapes1[i],
                                dtype=self.index_type_pd1,
                            )
                            for i in range(len(self.indices_shapes1))
                        ]
                    )
                else:
                    indices = tuple(
                        [
                            paddle.static.data(
                                name="indice" + str(i),
                                shape=self.indices_shapes[i],
                                dtype=self.index_type_pd,
                            )
                            for i in range(len(self.indices_shapes))
                        ]
                    )

                out = paddle.index_get(x, indices)
                exe = paddle.static.Executor(place=place)
                feed_list = {"x": self.x_np}
                for i in range(len(indices)):
                    feed_list.update({"indice" + str(i): self.indices_np[i]})
                pd_res = exe.run(
                    feed=feed_list,
                    fetch_list=[out],
                )
                ref_res = compute_index_get_ref(self.x_np, self.indices_np)
                np.testing.assert_allclose(ref_res, pd_res[0], atol=1e-7)
        paddle.disable_static()


# === Basic shapes and dtypes ===


class TestIndexGetAPI0(TestIndexGetAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.int64
        self.x_shape = (100, 110)
        self.indices_shapes = [(21,), (21,)]
        self.dtype_pd = "float64"
        self.index_type_pd = "int64"


class TestIndexGetAPI1(TestIndexGetAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.int64
        self.x_shape = (110, 42, 56, 56)
        self.indices_shapes = ((16, 16), (16, 16), (1, 16), (1, 16))
        self.dtype_pd = "float64"
        self.index_type_pd = "int64"


class TestIndexGetAPI2(TestIndexGetAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.int64
        self.x_shape = (110, 42, 56, 56)
        self.indices_shapes = ((16, 16), (16, 16), (1, 16))
        self.dtype_pd = "float64"
        self.index_type_pd = "int64"


class TestIndexGetAPI3(TestIndexGetAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.bool_
        self.x_shape = (110, 94)
        self.indices_shapes = [(110, 94)]
        self.dtype_pd = "float64"
        self.index_type_pd = "bool"


class TestIndexGetAPI4(TestIndexGetAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.bool_
        self.x_shape = (110, 94)
        self.indices_shapes = [(110,)]
        self.dtype_pd = "float64"
        self.index_type_pd = "bool"


class TestIndexGetAPI5(TestIndexGetAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float32
        self.index_type_np = np.int32
        self.x_shape = (100, 110)
        self.indices_shapes = [(21,), (21,)]
        self.dtype_pd = "float32"
        self.index_type_pd = "int32"


class TestIndexGetAPI6(TestIndexGetAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float32
        self.index_type_np = np.int64
        self.x_shape = (110, 42, 56, 56)
        self.indices_shapes = ((16, 16), (16, 16), (1, 16))
        self.dtype_pd = "float32"
        self.index_type_pd = "int64"


class TestIndexGetAPI7(TestIndexGetAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float16
        self.index_type_np = np.int32
        self.x_shape = (100, 110)
        self.indices_shapes = [(21,), (21,)]
        self.dtype_pd = "float16"
        self.index_type_pd = "int32"


class TestIndexGetAPI8(TestIndexGetAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.int32
        self.index_type_np = np.int32
        self.x_shape = (100, 110)
        self.indices_shapes = [(21,), (21,)]
        self.dtype_pd = "int32"
        self.index_type_pd = "int32"


class TestIndexGetAPI9(TestIndexGetAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.int64
        self.index_type_np = np.int32
        self.x_shape = (100, 110)
        self.indices_shapes = [(21,), (21,)]
        self.dtype_pd = "int64"
        self.index_type_pd = "int32"


class TestIndexGetAPI10(TestIndexGetAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.bool_
        self.index_type_np = np.int32
        self.x_shape = (100, 110)
        self.indices_shapes = [(21,), (21,)]
        self.dtype_pd = "bool"
        self.index_type_pd = "int32"


# === 1-D x with 1-D indices ===


class TestIndexGetAPI11(TestIndexGetAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.int64
        self.x_shape = (100,)
        self.indices_shapes = [(21,)]
        self.dtype_pd = "float64"
        self.index_type_pd = "int64"


class TestIndexGetAPI12(TestIndexGetAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float32
        self.index_type_np = np.int32
        self.x_shape = (50,)
        self.indices_shapes = [(30,)]
        self.dtype_pd = "float32"
        self.index_type_pd = "int32"


# === Broadcasting indices ===


class TestIndexGetAPI13(TestIndexGetAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.int64
        self.x_shape = (110, 42, 56, 56)
        self.indices_shapes = ((16, 1), (1, 16), (1, 1))
        self.dtype_pd = "float64"
        self.index_type_pd = "int64"


class TestIndexGetAPI14(TestIndexGetAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.int64
        self.x_shape = (110, 42, 56, 56)
        self.indices_shapes = ((16, 16), (16, 16), (16, 16))
        self.dtype_pd = "float64"
        self.index_type_pd = "int64"


# === Negative indices ===


class TestIndexGetAPI15(TestIndexGetAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.int64
        self.x_shape = (10, 10)
        self.indices_shapes = [(5,), (5,)]
        self.dtype_pd = "float64"
        self.index_type_pd = "int64"

    def setUp(self):
        super().setUp()
        self.indices_np = (
            np.array([0, -1, 2, -3, 4]),
            np.array([-1, 0, -2, 3, 1]),
        )


# === All-false bool indices ===


class TestIndexGetAPI16(TestIndexGetAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.int32
        self.x_shape = (100, 110)
        self.indices_shapes = [(21,), (21,)]
        self.dtype_pd = "float64"
        self.index_type_pd = "int32"
        self.is_all_false = True


# === Bool indices with True entries ===


class TestIndexGetAPI17(TestIndexGetAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.bool_
        self.x_shape = (44, 94)
        self.indices_shapes = [(44,)]
        self.dtype_pd = "float64"
        self.index_type_pd = "bool"

    def setUp(self):
        super().setUp()
        self.indices_np = (
            np.array([True, False, True, True] * 11, dtype=np.bool_),
        )


# === bool indices (all False) ===


class TestIndexGetAPI18(TestIndexGetAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.bool_
        self.x_shape = (100, 110)
        self.indices_shapes = [(100,)]
        self.dtype_pd = "float64"
        self.index_type_pd = "bool"
        self.is_all_false = True


# === Mixed indices: int + bool ===


class TestIndexGetAPIMixedIndices(TestIndexGetAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.int32
        self.x_shape = (110, 42, 32, 56)
        self.indices_shapes = ((16, 16), (16, 16))
        self.dtype_pd = "float64"
        self.index_type_pd = "int32"

        self.mixed_indices = True
        self.index_type_np1 = np.bool_
        self.indices_shapes1 = [(32,)]
        self.index_type_pd1 = "bool"


# === 3D x with 3D indices ===


class TestIndexGetAPI19(TestIndexGetAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.int64
        self.x_shape = (30, 40, 50)
        self.indices_shapes = ((10, 10), (10, 10), (10, 10))
        self.dtype_pd = "float64"
        self.index_type_pd = "int64"


# === Scalar-like output (single element indexing) ===


class TestIndexGetAPI20(TestIndexGetAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.int64
        self.x_shape = (10, 20)
        self.indices_shapes = [(1,), (1,)]
        self.dtype_pd = "float64"
        self.index_type_pd = "int64"


# === Inplace API ===


class TestIndexGetInplaceAPI(unittest.TestCase):
    def setUp(self):
        self.init_dtype_type()
        self.setPlace()
        self.x_np = np.random.random(self.x_shape).astype(self.dtype_np)
        self.indices_np = gen_indices_np(
            self.x_shape, self.indices_shapes, self.index_type_np, False
        )

    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.int64
        self.x_shape = (100, 110)
        self.indices_shapes = [(21,), (21,)]
        self.dtype_pd = "float64"
        self.index_type_pd = "int64"

    def setPlace(self):
        self.place = get_devices()

    def test_dygraph_forward(self):
        paddle.disable_static()
        for place in self.place:
            paddle.device.set_device(place)
            self.x_pd = paddle.to_tensor(self.x_np, dtype=self.dtype_pd)
            self.indices_pd = [
                paddle.to_tensor(indice, dtype=self.index_type_pd)
                for indice in self.indices_np
            ]
            self.indices_pd = tuple(self.indices_pd)
            ref_res = compute_index_get_ref(self.x_np, self.indices_np)
            x_pd_bk = self.x_pd.clone()
            pd_res = x_pd_bk.index_get_(self.indices_pd)
            np.testing.assert_allclose(ref_res, pd_res.numpy(), atol=1e-7)
            np.testing.assert_allclose(ref_res, x_pd_bk.numpy(), atol=1e-7)


class TestIndexGetInplaceAPI1(TestIndexGetInplaceAPI):
    def init_dtype_type(self):
        self.dtype_np = np.float32
        self.index_type_np = np.int32
        self.x_shape = (50, 60)
        self.indices_shapes = [(15,), (15,)]
        self.dtype_pd = "float32"
        self.index_type_pd = "int32"


# === Backward tests ===


class TestIndexGetAPIBackward(unittest.TestCase):
    def setUp(self):
        self.setPlace()

    def setPlace(self):
        self.place = get_devices()

    def test_backward(self):
        paddle.disable_static()
        for place in self.place:
            paddle.device.set_device(place)
            x = paddle.ones(shape=[16, 21], dtype="float64")
            ix1 = paddle.to_tensor([0, 1, 2, 3], dtype="int64")
            ix2 = paddle.to_tensor([0, 1, 2, 3], dtype="int64")
            x.stop_gradient = False
            out = paddle.index_get(x, (ix1, ix2))

            dx = paddle.grad(
                outputs=[out],
                inputs=[x],
                create_graph=False,
                retain_graph=True,
            )[0]
            ref_dx = np.zeros(shape=[16, 21], dtype=np.float64)
            ref_dx[ix1, ix2] = 1.0

            np.testing.assert_allclose(ref_dx, dx.numpy(), atol=1e-7)

    def test_backward_broadcast(self):
        paddle.disable_static()
        for place in self.place:
            paddle.device.set_device(place)
            x = paddle.ones(shape=[16, 21], dtype="float64")
            ix1 = paddle.to_tensor([[0, 1], [2, 3]], dtype="int64")
            ix2 = paddle.to_tensor([[0, 1], [2, 3]], dtype="int64")
            x.stop_gradient = False
            out = paddle.index_get(x, (ix1, ix2))

            dx = paddle.grad(
                outputs=[out],
                inputs=[x],
                create_graph=False,
                retain_graph=True,
            )[0]
            ref_dx = np.zeros(shape=[16, 21], dtype=np.float64)
            ref_dx[ix1.numpy(), ix2.numpy()] = 1.0

            np.testing.assert_allclose(ref_dx, dx.numpy(), atol=1e-7)

    def test_backward_1d(self):
        paddle.disable_static()
        for place in self.place:
            paddle.device.set_device(place)
            x = paddle.ones(shape=[100], dtype="float64")
            ix = paddle.to_tensor([0, 10, 20, 30, 40], dtype="int64")
            x.stop_gradient = False
            out = paddle.index_get(x, (ix,))

            dx = paddle.grad(
                outputs=[out],
                inputs=[x],
                create_graph=False,
                retain_graph=True,
            )[0]
            ref_dx = np.zeros(shape=[100], dtype=np.float64)
            ref_dx[ix.numpy()] = 1.0

            np.testing.assert_allclose(ref_dx, dx.numpy(), atol=1e-7)

    def test_backward_bool_indices(self):
        paddle.disable_static()
        for place in self.place:
            paddle.device.set_device(place)
            x = paddle.ones(shape=[16, 21], dtype="float64")
            ix = paddle.to_tensor([True, False, True, True] * 4, dtype="bool")[
                :16
            ]
            x.stop_gradient = False
            out = paddle.index_get(x, (ix,))

            dx = paddle.grad(
                outputs=[out],
                inputs=[x],
                create_graph=False,
                retain_graph=True,
            )[0]
            ref_dx = np.zeros(shape=[16, 21], dtype=np.float64)
            ref_dx[ix.numpy()] = 1.0

            np.testing.assert_allclose(ref_dx, dx.numpy(), atol=1e-7)

    def test_backward_all_false_bool(self):
        paddle.disable_static()
        for place in self.place:
            paddle.device.set_device(place)
            x = paddle.ones(shape=[16, 21], dtype="float64")
            ix = paddle.zeros(shape=[16, 21], dtype="bool")
            x.stop_gradient = False
            out = paddle.index_get(x, (ix,))

            dx = paddle.grad(
                outputs=[out],
                inputs=[x],
                create_graph=False,
                retain_graph=True,
            )[0]
            ref_dx = np.zeros(shape=[16, 21], dtype=np.float64)

            np.testing.assert_allclose(ref_dx, dx.numpy(), atol=1e-7)

    def test_backward_in_static(self):
        paddle.enable_static()
        exe = paddle.static.Executor()
        train_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(train_program, startup_program):
            x = paddle.zeros((4, 2, 5))
            x.stop_gradient = False

            y = x + 1
            index = paddle.to_tensor([0, 1, 3])

            z = paddle.index_get(y, (index,))
            l = z.sum()
            if paddle.framework.in_pir_mode():
                grads = paddle.autograd.ir_backward.grad(l, [x])
                x_grad = grads[0]
            else:
                paddle.static.append_backward(l)
                x_grad = x.grad_name

            res = exe.run(fetch_list=[z, x_grad])

            expected_z = np.ones((3, 2, 5))
            expected_x_grad = np.zeros((4, 2, 5))
            expected_x_grad[[0, 1, 3]] = 1.0

            np.testing.assert_allclose(expected_z, res[0])
            np.testing.assert_allclose(expected_x_grad, res[1])
        paddle.disable_static()


# === Trailing dims (fewer indices than x dims) ===


class TestIndexGetAPI21(TestIndexGetAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.int64
        self.x_shape = (10, 20, 30)
        self.indices_shapes = [(5,), (5,)]
        self.dtype_pd = "float64"
        self.index_type_pd = "int64"


class TestIndexGetAPI22(TestIndexGetAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.int64
        self.x_shape = (10, 20, 30, 40)
        self.indices_shapes = [(5,)]
        self.dtype_pd = "float64"
        self.index_type_pd = "int64"


# === Zero-size dims ===


class TestIndexGetAPI_ZeroSize(unittest.TestCase):
    def setUp(self):
        self.init_dtype_type()
        self.setPlace()

    def init_dtype_type(self):
        self.dtype_np = np.float32
        self.index_type_np = np.int64
        self.x_shape = (10, 0)
        self.indices_shapes = [[10]]
        self.dtype_pd = paddle.float32
        self.index_type_pd = paddle.int64

    def setPlace(self):
        self.place = get_devices()
        if self.dtype_np is np.float16 and "cpu" in self.place:
            self.place.remove("cpu")

    def test_dygraph_forward(self):
        paddle.disable_static()
        for place in self.place:
            paddle.device.set_device(place)
            x_pd = paddle.randn(self.x_shape, dtype=self.dtype_pd)
            x_np = x_pd.numpy()
            x_pd.stop_gradient = False
            indices_pd = [
                paddle.randint(0, 1, indices_shape).astype(
                    dtype=self.index_type_pd
                )
                for indices_shape in self.indices_shapes
            ]
            indices_np = [item.numpy() for item in indices_pd]
            indices_pd = tuple(indices_pd)
            ref_res = compute_index_get_ref(x_np, indices_np)
            pd_res = paddle.index_get(x_pd, indices_pd)
            np.testing.assert_allclose(ref_res, pd_res.numpy(), atol=1e-7)

            pd_res.sum().backward()
            np.testing.assert_allclose(x_pd.grad.shape, x_pd.shape)


# === Tensor method API ===


class TestTensorIndexGet(unittest.TestCase):
    def setUp(self):
        self.setPlace()

    def setPlace(self):
        self.place = get_devices()

    def test_tensor_method(self):
        paddle.disable_static()
        for place in self.place:
            paddle.device.set_device(place)
            x_np = np.random.random((10, 20)).astype(np.float64)
            x = paddle.to_tensor(x_np)
            ix1 = paddle.to_tensor([0, 2, 4], dtype="int64")
            ix2 = paddle.to_tensor([1, 3, 5], dtype="int64")

            result1 = paddle.index_get(x, (ix1, ix2))
            result2 = x.index_get((ix1, ix2))

            np.testing.assert_allclose(
                result1.numpy(), result2.numpy(), atol=1e-7
            )
            np.testing.assert_allclose(
                result1.numpy(),
                x_np[ix1.numpy(), ix2.numpy()],
                atol=1e-7,
            )


# === Compatibility tests ===


class TestIndexGetAPI_Compatibility(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        paddle.enable_static()
        self.shape = [5, 6]
        self.dtype = 'float32'
        self.init_data()

    def init_data(self):
        self.np_input = np.random.randint(0, 10, self.shape).astype(self.dtype)
        self.idx0 = np.array([0, 2], dtype='int64')
        self.idx1 = np.array([1, 3], dtype='int64')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()

        x = paddle.to_tensor(self.np_input, dtype=self.dtype)
        idx0_t = paddle.to_tensor(self.idx0, dtype='int64')
        idx1_t = paddle.to_tensor(self.idx1, dtype='int64')
        indices_t = (idx0_t, idx1_t)

        paddle_dygraph_out = []

        # 1) position args
        out1 = paddle.index_get(x, indices_t)
        paddle_dygraph_out.append(out1)

        # 2) paddle-style kwargs
        out2 = paddle.index_get(x=x, indices=indices_t)
        paddle_dygraph_out.append(out2)

        # 3) torch-style kwarg name 'input'
        out3 = paddle.index_get(input=x, indices=indices_t)
        paddle_dygraph_out.append(out3)

        ref_out = compute_index_get_ref(self.np_input, (self.idx0, self.idx1))

        np.testing.assert_allclose(ref_out, paddle_dygraph_out[0].numpy())
        np.testing.assert_allclose(ref_out, paddle_dygraph_out[1].numpy())
        np.testing.assert_allclose(ref_out, paddle_dygraph_out[2].numpy())

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)
            idx0_t = paddle.static.data(name="idx0", shape=[2], dtype='int64')
            idx1_t = paddle.static.data(name="idx1", shape=[2], dtype='int64')

            indices_t = (idx0_t, idx1_t)

            # position args
            out1 = paddle.index_get(x, indices_t)
            # paddle kwargs
            out2 = paddle.index_get(x=x, indices=indices_t)
            # torch-style kwarg name 'input'
            out3 = paddle.index_get(input=x, indices=indices_t)

            exe = paddle.static.Executor(paddle.CPUPlace())
            fetches = exe.run(
                feed={
                    "x": self.np_input,
                    "idx0": self.idx0,
                    "idx1": self.idx1,
                },
                fetch_list=[out1, out2, out3],
            )

            ref_out = compute_index_get_ref(
                self.np_input,
                (self.idx0, self.idx1),
            )

            for out in fetches:
                np.testing.assert_allclose(out, ref_out)


# === Non-contiguous / stride tests ===


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device()),
    "core is not compiled with CUDA",
)
class TestIndexGetOp_Stride(unittest.TestCase):
    def setUp(self):
        self.is_all_false = False
        self.init_dtype_type()
        self.setPlace()
        self.x_np = np.random.random(self.x_shape).astype(self.dtype_np)
        self.x_trans_np = np.transpose(self.x_np, self.perm)
        self.indices_np = gen_indices_np(
            self.x_shape,
            self.indices_shapes,
            self.index_type_np,
            self.is_all_false,
        )

    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.int64
        self.x_shape = (100, 110)
        self.indices_shapes = [(21,), (21,)]
        self.perm = [1, 0]
        self.dtype_pd = "float64"
        self.index_type_pd = "int64"

    def setPlace(self):
        self.place = get_device_place()

    def test_dygraph_forward(self):
        paddle.disable_static()
        paddle.device.set_device(self.place)
        self.x_pd = paddle.to_tensor(self.x_np, dtype=self.dtype_pd)
        self.x_trans_pd = paddle.to_tensor(self.x_trans_np, dtype=self.dtype_pd)
        self.indices_pd = [
            paddle.to_tensor(indice) for indice in self.indices_np
        ]
        self.indices_pd = tuple(self.indices_pd)
        self.x_non_conti = paddle.transpose(self.x_trans_pd, self.perm)
        ref_res = compute_index_get_ref(self.x_np, self.indices_np)
        pd_res = paddle.index_get(self.x_non_conti, self.indices_pd)
        np.testing.assert_allclose(ref_res, pd_res.numpy(), atol=1e-7)


# === PIR mode symbolic shape inference tests ===
# These exercise IndexGetOpInferSymbolicShape in binary_infer_sym.cc


class TestIndexGetPIRSymbolicShape(unittest.TestCase):
    """Exercise symbolic shape inference in PIR mode (binary_infer_sym.cc)."""

    def _run_pir_static(self, x_np, indices_np, dtype, expected_shape):
        with paddle.pir_utils.IrGuard():
            program = paddle.static.Program()
            startup = paddle.static.Program()
            with paddle.static.program_guard(program, startup):
                x = paddle.static.data(name="x", shape=x_np.shape, dtype=dtype)
                x.stop_gradient = False
                feed = {"x": x_np}
                indices = []
                for i, idx_np in enumerate(indices_np):
                    name = f"idx{i}"
                    idx_dtype = str(idx_np.dtype)
                    if "bool" in idx_dtype:
                        idx_dtype = "bool"
                    else:
                        idx_dtype = "int64"
                    idx = paddle.static.data(
                        name=name, shape=idx_np.shape, dtype=idx_dtype
                    )
                    indices.append(idx)
                    feed[name] = idx_np
                out = paddle.index_get(x, tuple(indices))
            exe = paddle.static.Executor(paddle.CPUPlace())
            result = exe.run(program, feed=feed, fetch_list=[out])
            self.assertEqual(list(result[0].shape), list(expected_shape))

    def test_pir_int_indices(self):
        self._run_pir_static(
            x_np=np.random.randn(10, 20).astype("float32"),
            indices_np=[
                np.array([0, 2, 4, 6, 8], dtype="int64"),
                np.array([1, 3, 5, 7, 9], dtype="int64"),
            ],
            dtype="float32",
            expected_shape=[5],
        )

    def test_pir_bool_indices(self):
        self._run_pir_static(
            x_np=np.random.randn(10, 20).astype("float32"),
            indices_np=[
                np.array(
                    [
                        True,
                        False,
                        True,
                        False,
                        True,
                        False,
                        True,
                        False,
                        True,
                        False,
                    ],
                    dtype="bool",
                ),
            ],
            dtype="float32",
            expected_shape=[5, 20],
        )

    def test_pir_mixed_bool_int_indices(self):
        self._run_pir_static(
            x_np=np.random.randn(10, 20, 30).astype("float32"),
            indices_np=[
                np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype="int64"),
                np.array([True, False] * 10, dtype="bool"),
            ],
            dtype="float32",
            expected_shape=[10, 30],
        )

    def test_pir_bool_2d_indices(self):
        self._run_pir_static(
            x_np=np.random.randn(10, 20).astype("float32"),
            indices_np=[
                np.eye(10, 20, dtype="bool"),
            ],
            dtype="float32",
            expected_shape=[10],
        )

    def test_pir_trailing_dims(self):
        self._run_pir_static(
            x_np=np.random.randn(10, 20, 30).astype("float32"),
            indices_np=[
                np.array([0, 2, 4, 6, 8], dtype="int64"),
            ],
            dtype="float32",
            expected_shape=[5, 20, 30],
        )


# === Additional backward tests for cpu/index_get_grad_kernel.cc coverage ===


class TestIndexGetGradDtypeFloat32(unittest.TestCase):
    def test_backward_float32(self):
        paddle.disable_static()
        for place in [paddle.CPUPlace()]:
            paddle.device.set_device(place)
            x = paddle.ones(shape=[16, 21], dtype="float32")
            ix1 = paddle.to_tensor([0, 1, 2, 3], dtype="int64")
            ix2 = paddle.to_tensor([0, 1, 2, 3], dtype="int64")
            x.stop_gradient = False
            out = paddle.index_get(x, (ix1, ix2))

            dx = paddle.grad(outputs=[out], inputs=[x])[0]
            ref_dx = np.zeros(shape=[16, 21], dtype=np.float32)
            ref_dx[ix1.numpy(), ix2.numpy()] = 1.0

            np.testing.assert_allclose(ref_dx, dx.numpy(), atol=1e-7)


class TestIndexGetGradDtypeInt32(unittest.TestCase):
    def test_backward_int32(self):
        paddle.disable_static()
        for place in [paddle.CPUPlace()]:
            paddle.device.set_device(place)
            x = paddle.to_tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype="int32")
            ix = paddle.to_tensor([0, 2, 4, 6, 8], dtype="int64")
            x.stop_gradient = False
            out = paddle.index_get(x, (ix,))
            dx = paddle.grad(outputs=[out], inputs=[x])[0]
            ref_dx = np.zeros(shape=[10], dtype=np.int32)
            ref_dx[ix.numpy()] = 1
            np.testing.assert_allclose(ref_dx, dx.numpy())


class TestIndexGetGradTrailingDims(unittest.TestCase):
    def test_backward_trailing_dims(self):
        paddle.disable_static()
        for place in [paddle.CPUPlace()]:
            paddle.device.set_device(place)
            x = paddle.ones(shape=[10, 20, 30], dtype="float64")
            ix = paddle.to_tensor([0, 2, 4, 6, 8], dtype="int64")
            x.stop_gradient = False
            out = paddle.index_get(x, (ix,))
            dx = paddle.grad(outputs=[out], inputs=[x])[0]
            ref_dx = np.zeros(shape=[10, 20, 30], dtype=np.float64)
            ref_dx[ix.numpy()] = 1.0
            np.testing.assert_allclose(ref_dx, dx.numpy(), atol=1e-7)


class TestIndexGetGradBroadcastBool(unittest.TestCase):
    def test_backward_bool_broadcast(self):
        paddle.disable_static()
        for place in [paddle.CPUPlace()]:
            paddle.device.set_device(place)
            x = paddle.ones(shape=[16, 21], dtype="float64")
            ix = paddle.to_tensor([True, False] * 8, dtype="bool")
            x.stop_gradient = False
            out = paddle.index_get(x, (ix,))
            dx = paddle.grad(outputs=[out], inputs=[x])[0]
            ref_dx = np.zeros(shape=[16, 21], dtype=np.float64)
            ref_dx[ix.numpy()] = 1.0
            np.testing.assert_allclose(ref_dx, dx.numpy(), atol=1e-7)


class TestIndexGetGradMixedIndices(unittest.TestCase):
    def test_backward_mixed_bool_int(self):
        paddle.disable_static()
        for place in [paddle.CPUPlace()]:
            paddle.device.set_device(place)
            x = paddle.ones(shape=[10, 20, 30], dtype="float64")
            ix1 = paddle.to_tensor(
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype="int64"
            )
            ix2 = paddle.to_tensor([True, False] * 10, dtype="bool")
            x.stop_gradient = False
            out = paddle.index_get(x, (ix1, ix2))
            dx = paddle.grad(outputs=[out], inputs=[x])[0]
            ref_dx = np.zeros(shape=[10, 20, 30], dtype=np.float64)
            np.add.at(ref_dx, (ix1.numpy(), ix2.numpy()), 1.0)
            np.testing.assert_allclose(ref_dx, dx.numpy(), atol=1e-7)


# === Additional forward tests for cpu/index_get_kernel.cc coverage ===


class TestIndexGetAllFalseBoolTrailing(unittest.TestCase):
    """All-false bool with trailing dims → covers all-false early return path."""

    def test_all_false_bool_trailing(self):
        paddle.disable_static()
        for place in [paddle.CPUPlace()]:
            paddle.device.set_device(place)
            x = paddle.randn(shape=[10, 20, 30], dtype="float32")
            ix = paddle.zeros(shape=[10, 20], dtype="bool")
            out = paddle.index_get(x, (ix,))
            self.assertEqual(out.shape, [0, 30])
            self.assertEqual(out.dtype, paddle.float32)


class TestIndexGetTrailingDimsDtypes(unittest.TestCase):
    """Fewer indices than x dims with different dtypes."""

    def test_trailing_dims_int32(self):
        paddle.disable_static()
        for place in [paddle.CPUPlace()]:
            paddle.device.set_device(place)
            x_np = np.random.randn(8, 12, 16).astype("float32")
            x = paddle.to_tensor(x_np)
            ix = paddle.to_tensor([1, 3, 5, 7], dtype="int32")
            out = paddle.index_get(x, (ix,))
            ref = x_np[[1, 3, 5, 7]]
            np.testing.assert_allclose(ref, out.numpy(), atol=1e-7)

    def test_trailing_dims_bool_index(self):
        paddle.disable_static()
        for place in [paddle.CPUPlace()]:
            paddle.device.set_device(place)
            x_np = np.random.randn(8, 12, 16).astype("float64")
            x = paddle.to_tensor(x_np)
            ix_np = np.array(
                [True, False, True, True, False, True, False, False],
                dtype="bool",
            )
            ix = paddle.to_tensor(ix_np)
            out = paddle.index_get(x, (ix,))
            ref = x_np[ix_np]
            np.testing.assert_allclose(ref, out.numpy(), atol=1e-7)


class TestIndexGetComplexDtype(unittest.TestCase):
    """Test complex64/complex128 dtype coverage in forward kernel."""

    def test_complex64(self):
        paddle.disable_static()
        for place in [paddle.CPUPlace()]:
            paddle.device.set_device(place)
            x_np = np.random.randn(10, 20).astype(
                "float32"
            ) + 1j * np.random.randn(10, 20).astype("float32")
            x = paddle.to_tensor(x_np)
            ix1 = paddle.to_tensor([0, 2, 4, 6, 8], dtype="int64")
            ix2 = paddle.to_tensor([1, 3, 5, 7, 9], dtype="int64")
            out = paddle.index_get(x, (ix1, ix2))
            ref = x_np[ix1.numpy(), ix2.numpy()]
            np.testing.assert_allclose(ref, out.numpy(), atol=1e-7)

    def test_complex128(self):
        paddle.disable_static()
        for place in [paddle.CPUPlace()]:
            paddle.device.set_device(place)
            x_np = np.random.randn(10, 20).astype(
                "float64"
            ) + 1j * np.random.randn(10, 20).astype("float64")
            x = paddle.to_tensor(x_np)
            ix1 = paddle.to_tensor([0, 2, 4, 6, 8], dtype="int64")
            ix2 = paddle.to_tensor([1, 3, 5, 7, 9], dtype="int64")
            out = paddle.index_get(x, (ix1, ix2))
            ref = x_np[ix1.numpy(), ix2.numpy()]
            np.testing.assert_allclose(ref, out.numpy(), atol=1e-7)


class TestIndexGetZeroSizeGrad(unittest.TestCase):
    """Test zero-size output gradient path in backward kernel."""

    def test_zero_size_out_grad(self):
        paddle.disable_static()
        for place in [paddle.CPUPlace()]:
            paddle.device.set_device(place)
            x = paddle.randn(shape=[10, 20], dtype="float32")
            # All-false bool produces zero-size output
            ix = paddle.zeros(shape=[10, 20], dtype="bool")
            x.stop_gradient = False
            out = paddle.index_get(x, (ix,))
            self.assertEqual(out.shape[0], 0)
            dx = paddle.grad(outputs=[out], inputs=[x])[0]
            # Gradient should be all zeros with correct shape
            self.assertEqual(list(dx.shape), [10, 20])
            np.testing.assert_allclose(
                np.zeros([10, 20], dtype="float32"), dx.numpy(), atol=1e-7
            )


if __name__ == '__main__':
    unittest.main()
