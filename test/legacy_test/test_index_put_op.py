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

import copy
import os
import unittest

import numpy as np

import paddle


def compute_index_put_ref(x_np, indices_np, value_np, accumulate=False):
    if accumulate:
        x_np[indices_np] += value_np
        return x_np
    else:
        x_np[indices_np] = value_np
        return x_np


def raw_index_put(x, indices, value, accummulate):
    return paddle.index_put(x, indices, value, accummulate)


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


class TestIndexPutAPIBase(unittest.TestCase):
    def setUp(self):
        self.mixed_indices = False
        self.is_all_false = False
        self.init_dtype_type()
        self.setPlace()
        self.x_np = np.random.random(self.x_shape).astype(self.dtype_np)
        self.value_np = np.random.random(self.value_shape).astype(self.dtype_np)

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
        self.value_shape = (21,)
        self.dtype_pd = "float64"
        self.index_type_pd = "int64"
        self.accumulate = False

    def setPlace(self):
        self.place = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.is_compiled_with_cuda()
        ):
            self.place.append('cpu')
        if self.dtype_np is np.float16:
            self.place = []
        if paddle.is_compiled_with_cuda():
            self.place.append('gpu')

    def test_dygraph_forward(self):
        paddle.disable_static()
        for place in self.place:
            paddle.device.set_device(place)
            self.x_pd = paddle.to_tensor(self.x_np, dtype=self.dtype_pd)
            self.value_pd = paddle.to_tensor(self.value_np, dtype=self.dtype_pd)
            self.indices_pd = [
                paddle.to_tensor(indice) for indice in self.indices_np
            ]
            self.indices_pd = tuple(self.indices_pd)
            ref_res = compute_index_put_ref(
                self.x_np, self.indices_np, self.value_np, self.accumulate
            )
            pd_res = paddle.index_put(
                self.x_pd, self.indices_pd, self.value_pd, self.accumulate
            )
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
                value = paddle.static.data(
                    name="value", shape=self.value_shape, dtype=self.dtype_pd
                )

                out = paddle.index_put(x, indices, value, self.accumulate)
                exe = paddle.static.Executor(place=place)
                feed_list = {}
                feed_list.update({"x": self.x_np})
                for i in range(len(indices)):
                    feed_list.update({"indice" + str(i): self.indices_np[i]})
                feed_list.update({"value": self.value_np})
                pd_res = exe.run(
                    feed=feed_list,
                    fetch_list=[out],
                )
                ref_res = compute_index_put_ref(
                    self.x_np, self.indices_np, self.value_np, self.accumulate
                )
                np.testing.assert_allclose(ref_res, pd_res[0], atol=1e-7)


class TestIndexPutAPI0(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.int64
        self.x_shape = (100, 110)
        self.indices_shapes = [(21,), (21,)]
        self.value_shape = (21,)
        self.dtype_pd = "float64"
        self.index_type_pd = "int64"
        self.accumulate = True


class TestIndexPutAPI1(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.int64
        self.x_shape = (110, 42, 56, 56)
        self.indices_shapes = ((16, 16), (16, 16), (1, 16), (1, 16))
        self.value_shape = (16, 16)
        self.dtype_pd = "float64"
        self.index_type_pd = "int64"
        self.accumulate = False


class TestIndexPutAPI2(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.int64
        self.x_shape = (110, 42, 56, 56)
        self.indices_shapes = ((16, 16), (16, 16), (1, 16), (1, 16))
        self.value_shape = (16, 16)
        self.dtype_pd = "float64"
        self.index_type_pd = "int64"
        self.accumulate = True


class TestIndexPutAPI3(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.bool_
        self.x_shape = (110, 94)
        self.indices_shapes = [(110, 94)]
        self.value_shape = (5170,)
        self.dtype_pd = "float64"
        self.index_type_pd = "bool"
        self.accumulate = False


class TestIndexPutAPI4(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.bool_
        self.x_shape = (110, 94)
        self.indices_shapes = [(110, 94)]
        self.value_shape = (5170,)
        self.dtype_pd = "float64"
        self.index_type_pd = "bool"
        self.accumulate = True


class TestIndexPutAPI5(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.int64
        self.x_shape = (110, 42, 56, 56)
        self.indices_shapes = ((16, 16), (16, 16), (1, 16))
        self.value_shape = (16, 16, 56)
        self.dtype_pd = "float64"
        self.index_type_pd = "int64"
        self.accumulate = False


class TestIndexPutAPI6(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.int64
        self.x_shape = (110, 42, 56, 56)
        self.indices_shapes = ((16, 16), (16, 16), (1, 16))
        self.value_shape = (16, 16, 56)
        self.dtype_pd = "float64"
        self.index_type_pd = "int64"
        self.accumulate = True


class TestIndexPutAPI7(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.bool_
        self.x_shape = (110, 94)
        self.indices_shapes = [(110,)]
        self.value_shape = (55, 94)
        self.dtype_pd = "float64"
        self.index_type_pd = "bool"
        self.accumulate = False


class TestIndexPutAPI8(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.bool_
        self.x_shape = (110, 94)
        self.indices_shapes = [(110,)]
        self.value_shape = (55, 94)
        self.dtype_pd = "float64"
        self.index_type_pd = "bool"
        self.accumulate = True


class TestIndexPutAPI9(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.int64
        self.x_shape = (110, 42, 56, 56)
        self.indices_shapes = ((16, 16), (16, 16), (1, 16))
        self.value_shape = (56,)
        self.dtype_pd = "float64"
        self.index_type_pd = "int64"
        self.accumulate = False


class TestIndexPutAPI10(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.int64
        self.x_shape = (110, 42, 56, 56)
        self.indices_shapes = ((16, 16), (16, 16), (1, 16))
        self.value_shape = (56,)
        self.dtype_pd = "float64"
        self.index_type_pd = "int64"
        self.accumulate = True


class TestIndexPutAPI11(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.int64
        self.x_shape = (110, 42, 56, 56)
        self.indices_shapes = ((16, 16), (16, 16), (1, 16))
        self.value_shape = (1,)
        self.dtype_pd = "float64"
        self.index_type_pd = "int64"
        self.accumulate = False


class TestIndexPutAPI12(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.int64
        self.x_shape = (110, 42, 56, 56)
        self.indices_shapes = ((16, 16), (16, 16), (1, 16))
        self.value_shape = (1,)
        self.dtype_pd = "float64"
        self.index_type_pd = "int64"
        self.accumulate = True


class TestIndexPutAPI13(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.bool_
        self.x_shape = (44, 94)
        self.indices_shapes = [(44,)]
        self.value_shape = (94,)
        self.dtype_pd = "float64"
        self.index_type_pd = "bool"
        self.accumulate = False


class TestIndexPutAPI14(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.bool_
        self.x_shape = (44, 94)
        self.indices_shapes = [(44,)]
        self.value_shape = (94,)
        self.dtype_pd = "float64"
        self.index_type_pd = "bool"
        self.accumulate = True


class TestIndexPutAPI15(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.bool_
        self.x_shape = (44, 94)
        self.indices_shapes = [(44,)]
        self.value_shape = (1,)
        self.dtype_pd = "float64"
        self.index_type_pd = "bool"
        self.accumulate = False


class TestIndexPutAPI16(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.bool_
        self.x_shape = (44, 94)
        self.indices_shapes = [(44,)]
        self.value_shape = (1,)
        self.dtype_pd = "float64"
        self.index_type_pd = "bool"
        self.accumulate = True


class TestIndexPutAPI17(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.int32
        self.x_shape = (100, 110)
        self.indices_shapes = [(21,), (21,)]
        self.value_shape = (21,)
        self.dtype_pd = "float64"
        self.index_type_pd = "int32"
        self.accumulate = False


class TestIndexPutAPI18(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.int32
        self.x_shape = (100, 110)
        self.indices_shapes = [(21,), (21,)]
        self.value_shape = (21,)
        self.dtype_pd = "float64"
        self.index_type_pd = "int32"
        self.accumulate = True


class TestIndexPutAPI19(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float32
        self.index_type_np = np.int32
        self.x_shape = (100, 110)
        self.indices_shapes = [(21,), (21,)]
        self.value_shape = (21,)
        self.dtype_pd = "float32"
        self.index_type_pd = "int32"
        self.accumulate = False


class TestIndexPutAPI20(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float32
        self.index_type_np = np.int32
        self.x_shape = (100, 110)
        self.indices_shapes = [(21,), (21,)]
        self.value_shape = (21,)
        self.dtype_pd = "float32"
        self.index_type_pd = "int32"
        self.accumulate = True


class TestIndexPutAPI21(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float16
        self.index_type_np = np.int32
        self.x_shape = (100, 110)
        self.indices_shapes = [(21,), (21,)]
        self.value_shape = (21,)
        self.dtype_pd = "float16"
        self.index_type_pd = "int32"
        self.accumulate = False


class TestIndexPutAPI22(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float16
        self.index_type_np = np.int32
        self.x_shape = (100, 110)
        self.indices_shapes = [(21,), (21,)]
        self.value_shape = (21,)
        self.dtype_pd = "float16"
        self.index_type_pd = "int32"
        self.accumulate = True


class TestIndexPutAPI23(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.int32
        self.index_type_np = np.int32
        self.x_shape = (100, 110)
        self.indices_shapes = [(21,), (21,)]
        self.value_shape = (21,)
        self.dtype_pd = "int32"
        self.index_type_pd = "int32"
        self.accumulate = False


class TestIndexPutAPI24(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.int32
        self.index_type_np = np.int32
        self.x_shape = (100, 110)
        self.indices_shapes = [(21,), (21,)]
        self.value_shape = (21,)
        self.dtype_pd = "int32"
        self.index_type_pd = "int32"
        self.accumulate = True


class TestIndexPutAPI25(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.int64
        self.index_type_np = np.int32
        self.x_shape = (100, 110)
        self.indices_shapes = [(21,), (21,)]
        self.value_shape = (21,)
        self.dtype_pd = "int64"
        self.index_type_pd = "int32"
        self.accumulate = False


class TestIndexPutAPI26(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.int64
        self.index_type_np = np.int32
        self.x_shape = (100, 110)
        self.indices_shapes = [(21,), (21,)]
        self.value_shape = (21,)
        self.dtype_pd = "int64"
        self.index_type_pd = "int32"
        self.accumulate = True


class TestIndexPutAPI27(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.bool_
        self.index_type_np = np.int32
        self.x_shape = (100, 110)
        self.indices_shapes = [(21,), (21,)]
        self.value_shape = (21,)
        self.dtype_pd = "bool"
        self.index_type_pd = "int32"
        self.accumulate = False


class TestIndexPutAPI28(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.bool_
        self.index_type_np = np.int32
        self.x_shape = (100, 110)
        self.indices_shapes = [(21,), (21,)]
        self.value_shape = (21,)
        self.dtype_pd = "bool"
        self.index_type_pd = "int32"
        self.accumulate = True


class TestIndexPutAPI29(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.int32
        self.x_shape = (110, 42, 56, 56)
        self.indices_shapes = ((16, 16), (16, 16), (1, 16))
        self.value_shape = (16, 16, 56)
        self.dtype_pd = "float64"
        self.index_type_pd = "int32"
        self.accumulate = False


class TestIndexPutAPI30(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.int32
        self.x_shape = (110, 42, 56, 56)
        self.indices_shapes = ((16, 16), (16, 16), (1, 16))
        self.value_shape = (16, 16, 56)
        self.dtype_pd = "float64"
        self.index_type_pd = "int32"
        self.accumulate = True


class TestIndexPutAPI31(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.bool_
        self.index_type_np = np.int32
        self.x_shape = (100, 110)
        self.indices_shapes = [(21,), (21,)]
        self.value_shape = (21,)
        self.dtype_pd = "bool"
        self.index_type_pd = "int32"
        self.accumulate = False
        self.is_all_false = True


class TestIndexPutAPI32(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.bool_
        self.index_type_np = np.int32
        self.x_shape = (100, 110)
        self.indices_shapes = [(21,), (21,)]
        self.value_shape = (21,)
        self.dtype_pd = "bool"
        self.index_type_pd = "int32"
        self.accumulate = True
        self.is_all_false = True


class TestIndexPutInplaceAPI(unittest.TestCase):
    def setUp(self):
        self.init_dtype_type()
        self.setPlace()
        self.x_np = np.random.random(self.x_shape).astype(self.dtype_np)
        self.value_np = np.random.random(self.value_shape).astype(self.dtype_np)
        self.indices_np = gen_indices_np(
            self.x_shape, self.indices_shapes, self.index_type_np, False
        )

    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.int64
        self.x_shape = (100, 110)
        self.indices_shapes = [(21,), (21,)]
        self.value_shape = (21,)
        self.dtype_pd = "float64"
        self.index_type_pd = "int64"
        self.accumulate = False

    def setPlace(self):
        self.place = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.is_compiled_with_cuda()
        ):
            self.place.append('cpu')
        if paddle.is_compiled_with_cuda():
            self.place.append('gpu')

    def test_dygraph_forward(self):
        paddle.disable_static()
        for place in self.place:
            paddle.device.set_device(place)
            self.x_pd = paddle.to_tensor(self.x_np, dtype=self.dtype_pd)
            self.value_pd = paddle.to_tensor(self.value_np, dtype=self.dtype_pd)
            self.indices_pd = [
                paddle.to_tensor(indice, dtype=self.index_type_pd)
                for indice in self.indices_np
            ]
            self.indices_pd = tuple(self.indices_pd)
            ref_res = compute_index_put_ref(
                self.x_np, self.indices_np, self.value_np, self.accumulate
            )
            x_pd_bk = self.x_pd.clone()
            pd_res = paddle.index_put_(
                x_pd_bk, self.indices_pd, self.value_pd, self.accumulate
            )
            np.testing.assert_allclose(ref_res, pd_res.numpy(), atol=1e-7)
            np.testing.assert_allclose(ref_res, x_pd_bk.numpy(), atol=1e-7)


class TestIndexPutInplaceAPI1(TestIndexPutInplaceAPI):
    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.int64
        self.x_shape = (100, 110)
        self.indices_shapes = [(21,), (21,)]
        self.value_shape = (21,)
        self.dtype_pd = "float64"
        self.index_type_pd = "int64"
        self.accumulate = True


class TestIndexPutAPIBackward(unittest.TestCase):
    def setUp(self):
        self.setPlace()

    def setPlace(self):
        self.place = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.is_compiled_with_cuda()
        ):
            self.place.append('cpu')
        if paddle.is_compiled_with_cuda():
            self.place.append('gpu')

    def test_backward(self):
        paddle.disable_static()
        for place in self.place:
            paddle.device.set_device(place)
            value = paddle.ones(shape=[4], dtype="float64")
            x = paddle.ones(shape=[16, 21], dtype="float64")
            ix1 = paddle.to_tensor([0, 1, 2, 3], dtype="int64")
            ix2 = paddle.to_tensor([0, 1, 2, 3], dtype="int64")
            value.stop_gradient = False
            x.stop_gradient = False
            out = paddle.index_put(x, (ix1, ix2), value, False)

            dx, dvalue = paddle.grad(
                outputs=[out],
                inputs=[x, value],
                create_graph=False,
                retain_graph=True,
            )
            ref_dx = np.ones(shape=[16, 21], dtype=np.float64)
            ref_dx[ix1, ix2] = 0

            np.testing.assert_allclose(ref_dx, dx.numpy(), atol=1e-7)
            np.testing.assert_allclose(
                np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
                dvalue.numpy(),
                atol=1e-7,
            )

            out = paddle.index_put(x, (ix1, ix2), value, True)

            dx, dvalue = paddle.grad(
                outputs=[out],
                inputs=[x, value],
                create_graph=False,
                retain_graph=True,
            )
            ref_dx = np.ones(shape=[16, 21], dtype=np.float64)

            np.testing.assert_allclose(ref_dx, dx.numpy(), atol=1e-7)
            np.testing.assert_allclose(
                np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
                dvalue.numpy(),
                atol=1e-7,
            )

    def test_backward_scalarval(self):
        paddle.disable_static()
        for place in self.place:
            paddle.device.set_device(place)
            value = paddle.ones(shape=[1], dtype="float64")
            x = paddle.ones(shape=[16, 21], dtype="float64")
            ix1 = paddle.to_tensor([0, 1, 2, 3], dtype="int64")
            ix2 = paddle.to_tensor([0, 1, 2, 3], dtype="int64")
            value.stop_gradient = False
            x.stop_gradient = False
            out = paddle.index_put(x, (ix1, ix2), value, False)

            dx, dvalue = paddle.grad(
                outputs=[out],
                inputs=[x, value],
                create_graph=False,
                retain_graph=True,
            )
            ref_dx = np.ones(shape=[16, 21], dtype=np.float64)
            ref_dx[ix1, ix2] = 0

            np.testing.assert_allclose(ref_dx, dx.numpy(), atol=1e-7)
            np.testing.assert_allclose(
                np.array([4.0], dtype=np.float64), dvalue.numpy(), atol=1e-7
            )

            out = paddle.index_put(x, (ix1, ix2), value, True)

            dx, dvalue = paddle.grad(
                outputs=[out],
                inputs=[x, value],
                create_graph=False,
                retain_graph=True,
            )
            ref_dx = np.ones(shape=[16, 21], dtype=np.float64)

            np.testing.assert_allclose(ref_dx, dx.numpy(), atol=1e-7)
            np.testing.assert_allclose(
                np.array([4.0], dtype=np.float64), dvalue.numpy(), atol=1e-7
            )

    def test_backward_broadcastvalue(self):
        paddle.disable_static()
        for place in self.place:
            paddle.device.set_device(place)
            value = paddle.ones(shape=[2], dtype="float64")
            x = paddle.ones(shape=[16, 21], dtype="float64")
            ix1 = paddle.to_tensor([[0, 1], [2, 3]], dtype="int64")
            ix2 = paddle.to_tensor([[0, 1], [2, 3]], dtype="int64")
            value.stop_gradient = False
            x.stop_gradient = False
            out = paddle.index_put(x, (ix1, ix2), value, False)

            dx, dvalue = paddle.grad(
                outputs=[out],
                inputs=[x, value],
                create_graph=False,
                retain_graph=True,
            )
            ref_dx = np.ones(shape=[16, 21], dtype=np.float64)
            ref_dx[ix1, ix2] = 0

            np.testing.assert_allclose(ref_dx, dx.numpy(), atol=1e-7)
            np.testing.assert_allclose(
                np.array([2.0, 2.0], dtype=np.float64),
                dvalue.numpy(),
                atol=1e-7,
            )

            out = paddle.index_put(x, (ix1, ix2), value, True)

            dx, dvalue = paddle.grad(
                outputs=[out],
                inputs=[x, value],
                create_graph=False,
                retain_graph=True,
            )
            ref_dx = np.ones(shape=[16, 21], dtype=np.float64)

            np.testing.assert_allclose(ref_dx, dx.numpy(), atol=1e-7)
            np.testing.assert_allclose(
                np.array([2.0, 2.0], dtype=np.float64),
                dvalue.numpy(),
                atol=1e-7,
            )

    def test_backward_broadcastvalue1(self):
        paddle.disable_static()
        for place in self.place:
            paddle.device.set_device(place)
            value = paddle.ones(shape=[1, 2], dtype="float64")
            x = paddle.ones(shape=[16, 21], dtype="float64")
            ix1 = paddle.to_tensor([[0, 1], [2, 3]], dtype="int64")
            ix2 = paddle.to_tensor([[0, 1], [2, 3]], dtype="int64")
            value.stop_gradient = False
            x.stop_gradient = False
            out = paddle.index_put(x, (ix1, ix2), value, False)

            dx, dvalue = paddle.grad(
                outputs=[out],
                inputs=[x, value],
                create_graph=False,
                retain_graph=True,
            )
            ref_dx = np.ones(shape=[16, 21], dtype=np.float64)
            ref_dx[ix1, ix2] = 0

            np.testing.assert_allclose(ref_dx, dx.numpy(), atol=1e-7)
            np.testing.assert_allclose(
                np.array([[2.0, 2.0]], dtype=np.float64),
                dvalue.numpy(),
                atol=1e-7,
            )

            out = paddle.index_put(x, (ix1, ix2), value, True)

            dx, dvalue = paddle.grad(
                outputs=[out],
                inputs=[x, value],
                create_graph=False,
                retain_graph=True,
            )
            ref_dx = np.ones(shape=[16, 21], dtype=np.float64)

            np.testing.assert_allclose(ref_dx, dx.numpy(), atol=1e-7)
            np.testing.assert_allclose(
                np.array([[2.0, 2.0]], dtype=np.float64),
                dvalue.numpy(),
                atol=1e-7,
            )

    def test_backward_broadcastvalue2(self):
        paddle.disable_static()
        for place in self.place:
            paddle.device.set_device(place)
            value = paddle.ones(shape=[2, 1], dtype="float64")
            x = paddle.ones(shape=[16, 21], dtype="float64")
            ix1 = paddle.to_tensor([[0, 1], [2, 3]], dtype="int64")
            ix2 = paddle.to_tensor([[0, 1], [2, 3]], dtype="int64")
            value.stop_gradient = False
            x.stop_gradient = False
            out = paddle.index_put(x, (ix1, ix2), value, False)

            dx, dvalue = paddle.grad(
                outputs=[out],
                inputs=[x, value],
                create_graph=False,
                retain_graph=True,
            )
            ref_dx = np.ones(shape=[16, 21], dtype=np.float64)
            ref_dx[ix1, ix2] = 0

            np.testing.assert_allclose(ref_dx, dx.numpy(), atol=1e-7)
            np.testing.assert_allclose(
                np.array([[2.0], [2.0]], dtype=np.float64),
                dvalue.numpy(),
                atol=1e-7,
            )

            out = paddle.index_put(x, (ix1, ix2), value, True)

            dx, dvalue = paddle.grad(
                outputs=[out],
                inputs=[x, value],
                create_graph=False,
                retain_graph=True,
            )
            ref_dx = np.ones(shape=[16, 21], dtype=np.float64)

            np.testing.assert_allclose(ref_dx, dx.numpy(), atol=1e-7)
            np.testing.assert_allclose(
                np.array([[2.0], [2.0]], dtype=np.float64),
                dvalue.numpy(),
                atol=1e-7,
            )

    def test_backward_all_false_bool_indice(self):
        paddle.disable_static()
        for place in self.place:
            paddle.device.set_device(place)
            value = paddle.ones(shape=[2, 1], dtype="float64")
            x = paddle.ones(shape=[16, 21], dtype="float64")
            ix = paddle.zeros(shape=[16, 21], dtype="bool")

            value.stop_gradient = False
            x.stop_gradient = False
            out = paddle.index_put(x, (ix,), value, False)

            dx, dvalue = paddle.grad(
                outputs=[out],
                inputs=[x, value],
                create_graph=False,
                retain_graph=True,
            )
            ref_dx = np.ones(shape=[16, 21], dtype=np.float64)

            np.testing.assert_allclose(ref_dx, dx.numpy(), atol=1e-7)
            np.testing.assert_allclose(
                np.array([[0.0], [0.0]], dtype=np.float64),
                dvalue.numpy(),
                atol=1e-7,
            )

            out = paddle.index_put(x, (ix,), value, True)

            dx, dvalue = paddle.grad(
                outputs=[out],
                inputs=[x, value],
                create_graph=False,
                retain_graph=True,
            )
            ref_dx = np.ones(shape=[16, 21], dtype=np.float64)

            np.testing.assert_allclose(ref_dx, dx.numpy(), atol=1e-7)
            np.testing.assert_allclose(
                np.array([[0.0], [0.0]], dtype=np.float64),
                dvalue.numpy(),
                atol=1e-7,
            )

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

            value = paddle.ones((5,))
            value.stop_gradient = False

            z = paddle.index_put(y, (index,), value)
            l = z.sum()
            if paddle.framework.in_pir_mode():
                grads = paddle.autograd.ir_backward.grad(l, [x, value])
                x_grad = grads[0]
                value_grad = grads[1]
            else:
                paddle.static.append_backward(l)
                x_grad = x.grad_name
                value_grad = value.grad_name

            res = exe.run(fetch_list=[z, x_grad, value_grad])

            expected_z = np.ones((4, 2, 5))
            expected_z[[0, 1, 3]] = np.ones((5,))

            expected_x_grad = np.ones((4, 2, 5))
            expected_x_grad[[0, 1, 3]] = 0

            expected_v_grad = np.ones((5,)) * 3 * 2

            np.testing.assert_allclose(expected_z, res[0])
            np.testing.assert_allclose(expected_x_grad, res[1])
            np.testing.assert_allclose(expected_v_grad, res[2])
        paddle.disable_static()


class TestIndexPutAPIMixedIndices(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.int32
        self.x_shape = (110, 42, 32, 56)
        self.indices_shapes = ((16, 16), (16, 16))
        self.value_shape = (16, 16, 56)
        self.dtype_pd = "float64"
        self.index_type_pd = "int32"
        self.accumulate = False

        self.mixed_indices = True
        self.index_type_np1 = np.bool_
        self.indices_shapes1 = [(32,)]
        self.index_type_pd1 = "bool"


class TestIndexPutAPIMixedIndices1(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.int32
        self.x_shape = (110, 42, 32, 56)
        self.indices_shapes = ((16, 16), (16, 16))
        self.value_shape = (16, 16, 56)
        self.dtype_pd = "float64"
        self.index_type_pd = "int32"
        self.accumulate = True

        self.mixed_indices = True
        self.index_type_np1 = np.bool_
        self.indices_shapes1 = [(32,)]
        self.index_type_pd1 = "bool"


if __name__ == '__main__':
    unittest.main()
