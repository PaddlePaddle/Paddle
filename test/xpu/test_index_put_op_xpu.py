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
import unittest

import numpy as np
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test import convert_float_to_uint16, convert_uint16_to_float
from op_test_xpu import XPUOpTest

import paddle

paddle.enable_static()


def compute_index_put_ref(x_np, indices_np, value_np, accumulate=False):
    if accumulate:
        x_np[indices_np] += value_np
        return x_np
    else:
        x_np[indices_np] = value_np
        return x_np


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


class XPUTestIndexPut(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = "index_put"
        self.use_dynamic_create_class = False

    class TestXPUIndexPutOp(XPUOpTest):
        def setUp(self):
            self.op_type = "index_put"
            self.dtype = self.in_type
            self.mixed_indices = False
            self.is_all_false = False
            self.place = paddle.XPUPlace(0)

            self.set_case()
            self.init_data()

        def set_case(self):
            self.index_dtype = np.int64
            self.x_shape = (100, 110)
            self.indices_shapes = [(21,), (21,)]
            self.value_shape = (21,)
            self.accumulate = False

        def init_data(self):
            x_np = ((np.random.random(self.x_shape) - 0.5) * 10.0).astype(
                "float32"
            )
            value_np = (
                (np.random.random(self.value_shape) - 0.5) * 10.0
            ).astype("float32")

            if self.dtype == np.uint16:
                x_np = convert_float_to_uint16(x_np)
                value_np = convert_float_to_uint16(value_np)
            else:
                x_np = x_np.astype(self.dtype)
                value_np = value_np.astype(self.dtype)

            if self.mixed_indices:
                tmp_indices_np1 = gen_indices_np(
                    self.x_shape,
                    self.indices_shapes,
                    self.index_dtype,
                    self.is_all_false,
                )
                tmp_indices_np2 = gen_indices_np(
                    self.x_shape,
                    self.indices_shapes1,
                    self.index_dtype1,
                    self.is_all_false,
                )
                self.indices_np = tuple(
                    list(tmp_indices_np1) + list(tmp_indices_np2)
                )
            else:
                self.indices_np = gen_indices_np(
                    self.x_shape,
                    self.indices_shapes,
                    self.index_dtype,
                    self.is_all_false,
                )

            indices_names = self.get_indices_names()
            indices_name_np = []
            for index_name, index_np in zip(indices_names, self.indices_np):
                indices_name_np.append((index_name, index_np))

            self.inputs = {
                'x': x_np,
                'indices': indices_name_np,
                'value': value_np,
            }

            self.attrs = {'accumulate': self.accumulate}
            if self.is_all_false:
                out_np = x_np
            else:
                if self.dtype == np.uint16:
                    out_np = compute_index_put_ref(
                        convert_uint16_to_float(copy.deepcopy(x_np)),
                        self.indices_np,
                        convert_uint16_to_float(value_np),
                        self.accumulate,
                    )
                    out_np = convert_float_to_uint16(out_np)
                else:
                    out_np = compute_index_put_ref(
                        copy.deepcopy(x_np),
                        self.indices_np,
                        value_np,
                        self.accumulate,
                    )
            self.outputs = {'out': out_np}

        def get_indices_names(self):
            indices_names = []
            for i in range(len(self.indices_np)):
                indices_names.append(f"index_{i}")
            return indices_names

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            self.check_grad_with_place(self.place, ["x", "value"], "out")

    class TestXPUIndexPut1(TestXPUIndexPutOp):
        def set_case(self):
            self.index_dtype = np.int64
            self.x_shape = (48, 26, 56)
            self.indices_shapes = [(16, 16), (16, 16), (1, 16)]
            self.value_shape = [16, 16]
            self.accumulate = False

    class TestXPUIndexPut2(TestXPUIndexPutOp):
        def set_case(self):
            self.index_dtype = np.int64
            self.x_shape = (48, 26, 56)
            self.indices_shapes = [(16, 16), (16, 16), (1, 16)]
            self.value_shape = (16, 16)
            self.accumulate = True

    class TestXPUIndexPut3(TestXPUIndexPutOp):
        def set_case(self):
            self.index_dtype = np.bool_
            self.x_shape = (12, 94)
            self.indices_shapes = [(12, 94)]
            self.value_shape = (564,)
            self.accumulate = False

    class TestXPUIndexPut4(TestXPUIndexPutOp):
        def set_case(self):
            self.index_dtype = np.bool_
            self.x_shape = (11, 94)
            self.indices_shapes = [(11, 94)]
            self.value_shape = (564,)
            self.accumulate = True

    class TestXPUIndexPut5(TestXPUIndexPutOp):
        def set_case(self):
            self.index_dtype = np.int32
            self.x_shape = (17, 32, 26, 36)
            self.indices_shapes = ((8, 8), (8, 8), (1, 8))
            self.value_shape = (8, 8, 36)
            self.accumulate = False

    class TestXPUIndexPut6(TestXPUIndexPutOp):
        def set_case(self):
            self.index_dtype = np.int32
            self.x_shape = (17, 32, 26, 36)
            self.indices_shapes = ((8, 8), (8, 8), (1, 8))
            self.value_shape = (8, 8, 36)
            self.accumulate = True

    class TestXPUIndexPut7(TestXPUIndexPutOp):
        def set_case(self):
            self.index_dtype = np.bool_
            self.x_shape = (110, 94)
            self.indices_shapes = [(110,)]
            self.value_shape = (55, 94)
            self.accumulate = False
            self.is_all_false = True

    class TestXPUIndexPut8(TestXPUIndexPutOp):
        def set_case(self):
            self.index_dtype = np.bool_
            self.x_shape = (110, 94)
            self.indices_shapes = [(110,)]
            self.value_shape = (55, 94)
            self.accumulate = True

    class TestXPUIndexPut9(TestXPUIndexPutOp):
        def set_case(self):
            self.index_dtype = np.int64
            self.x_shape = (17, 32, 26, 36)
            self.indices_shapes = ((8, 8), (8, 8), (1, 8))
            self.value_shape = (36,)
            self.accumulate = False

    class TestXPUIndexPut10(TestXPUIndexPutOp):
        def set_case(self):
            self.index_dtype = np.int64
            self.x_shape = (17, 32, 26, 36)
            self.indices_shapes = ((8, 8), (8, 8), (8, 8))
            self.value_shape = (36,)
            self.accumulate = True

    class TestXPUIndexPut11(TestXPUIndexPutOp):
        def set_case(self):
            self.index_dtype = np.int64
            self.x_shape = (17, 32, 26, 36)
            self.indices_shapes = ((8, 8), (8, 8), (8, 8))
            self.value_shape = (1,)
            self.accumulate = False

    class TestXPUIndexPut12(TestXPUIndexPutOp):
        def set_case(self):
            self.index_dtype = np.int64
            self.x_shape = (17, 32, 26, 36)
            self.indices_shapes = ((8, 8), (8, 8), (1, 8))
            self.value_shape = (1,)
            self.accumulate = True

    class TestXPUIndexPut13(TestXPUIndexPutOp):
        def set_case(self):
            self.index_dtype = np.bool_
            self.x_shape = (44, 94)
            self.indices_shapes = [(44,)]
            self.value_shape = (94,)
            self.accumulate = False

    class TestXPUIndexPut14(TestXPUIndexPutOp):
        def set_case(self):
            self.index_dtype = np.bool_
            self.x_shape = (44, 94)
            self.indices_shapes = [(44,)]
            self.value_shape = (94,)
            self.accumulate = True

    class TestXPUIndexPut15(TestXPUIndexPutOp):
        def set_case(self):
            self.index_dtype = np.bool_
            self.x_shape = (44, 94)
            self.indices_shapes = [(44,)]
            self.value_shape = (1,)
            self.accumulate = False

    class TestXPUIndexPut16(TestXPUIndexPutOp):
        def set_case(self):
            self.index_dtype = np.bool_
            self.x_shape = (44, 94)
            self.indices_shapes = [(44,)]
            self.value_shape = (1,)
            self.accumulate = True

    class TestXPUIndexPut17(TestXPUIndexPutOp):
        def set_case(self):
            self.index_dtype = np.int32
            self.x_shape = (100, 110)
            self.indices_shapes = [(21,), (21,)]
            self.value_shape = (21,)
            self.accumulate = False

    class TestXPUIndexPut18(TestXPUIndexPutOp):
        def set_case(self):
            self.index_dtype = np.int32
            self.x_shape = (100, 110)
            self.indices_shapes = [(21,), (21,)]
            self.value_shape = (21,)
            self.accumulate = True

    class TestXPUIndexPutMixedIndices(TestXPUIndexPutOp):
        def set_case(self):
            self.index_dtype = np.int32
            self.x_shape = (17, 32, 16, 36)
            self.indices_shapes = ((8, 8), (8, 8))
            self.value_shape = (8, 8, 36)
            self.accumulate = False

            self.mixed_indices = True
            self.index_dtype1 = np.bool_
            self.indices_shapes1 = [(16,)]

    class TestXPUIndexPutMixedIndices1(TestXPUIndexPutOp):
        def set_case(self):
            self.index_dtype = np.int32
            self.x_shape = (17, 32, 16, 36)
            self.indices_shapes = ((8, 8), (8, 8))
            self.value_shape = (8, 8, 36)
            self.accumulate = True

            self.mixed_indices = True
            self.index_dtype1 = np.bool_
            self.indices_shapes1 = [(16,)]


supported_type = get_xpu_op_support_types("index_put")
for stype in supported_type:
    create_test_class(globals(), XPUTestIndexPut, stype)


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
        self.dtype_np = np.float32
        self.index_type_np = np.int64
        self.x_shape = (50, 55)
        self.indices_shapes = [(21,), (21,)]
        self.value_shape = (21,)
        self.dtype_pd = paddle.float32
        self.index_type_pd = paddle.int64
        self.accumulate = False

    def setPlace(self):
        self.place = ['xpu']

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
        paddle.enable_static()


class TestIndexPutInplaceAPI1(TestIndexPutInplaceAPI):
    def init_dtype_type(self):
        self.dtype_np = np.float32
        self.index_type_np = np.int64
        self.x_shape = (100, 110)
        self.indices_shapes = [(21,), (21,)]
        self.value_shape = (21,)
        self.dtype_pd = paddle.float32
        self.index_type_pd = paddle.int64
        self.accumulate = True


if __name__ == "__main__":
    unittest.main()
