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
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test_xpu import XPUOpTest

import paddle
from paddle import base

paddle.enable_static()


class XPUTestIndexSampleOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'index_sample'
        self.use_dynamic_create_class = False

    class TestIndexSampleOPBase(XPUOpTest):
        def setUp(self):
            self.place = paddle.XPUPlace(0)
            self.init_dtype()
            self.set_case()

        def set_case(self):
            self.op_type = 'index_sample'
            self.config()

            xnp = np.random.random(self.x_shape).astype(self.dtype)
            indexnp = np.random.randint(
                low=0, high=self.x_shape[1], size=self.index_shape
            ).astype(self.index_type)
            self.inputs = {'X': xnp, 'Index': indexnp}
            index_array = []
            for i in range(self.index_shape[0]):
                for j in indexnp[i]:
                    index_array.append(xnp[i, j])
            index_array = np.array(index_array).astype(self.dtype)
            out = np.reshape(index_array, self.index_shape)
            self.outputs = {'Out': out}

        def init_dtype(self):
            self.dtype = self.in_type

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            self.check_grad_with_place(self.place, ['X'], 'Out')

        def config(self):
            self.x_shape = (10, 20)
            self.index_shape = (10, 10)
            self.index_type = "int32"

    class XPUTestIndexSample1(TestIndexSampleOPBase):
        def config(self):
            self.x_shape = (100, 1)
            self.index_shape = (100, 1)
            self.index_type = "int32"

    class XPUTestIndexSample2(TestIndexSampleOPBase):
        def config(self):
            self.x_shape = (100, 1)
            self.index_shape = (100, 1)
            self.index_type = "int64"

    class XPUTestIndexSample3(TestIndexSampleOPBase):
        def config(self):
            self.x_shape = (10, 100)
            self.index_shape = (10, 10)
            self.index_type = "int64"

    class XPUTestIndexSample4(TestIndexSampleOPBase):
        def config(self):
            self.x_shape = (10, 100)
            self.index_shape = (10, 10)
            self.index_type = "int32"

    class XPUTestIndexSample5(TestIndexSampleOPBase):
        def config(self):
            self.x_shape = (10, 128)
            self.index_shape = (10, 64)
            self.index_type = "int64"

    class XPUTestIndexSample6(TestIndexSampleOPBase):
        def config(self):
            self.x_shape = (10, 128)
            self.index_shape = (10, 64)
            self.index_type = "int32"


class TestIndexSampleShape(unittest.TestCase):
    def test_shape(self):
        paddle.enable_static()
        # create x value
        x_shape = (2, 5)
        x_np = np.random.random(x_shape).astype('float32')

        # create index value
        index_shape = (2, 3)
        index_type = "int32"
        index_np = np.random.randint(
            low=0, high=x_shape[1], size=index_shape
        ).astype(index_type)

        x = paddle.static.data(name='x', shape=[-1, 5], dtype='float32')
        index = paddle.static.data(name='index', shape=[-1, 3], dtype='int32')
        output = paddle.index_sample(x=x, index=index)

        place = base.XPUPlace(0)
        exe = base.Executor(place=place)
        exe.run(base.default_startup_program())

        feed = {'x': x_np, 'index': index_np}
        res = exe.run(feed=feed, fetch_list=[output])


class TestIndexSampleDynamic(unittest.TestCase):
    def test_result(self):
        with base.dygraph.guard():
            x = paddle.to_tensor(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                ],
                dtype='float32',
            )
            index = paddle.to_tensor(
                [[0, 1, 2], [1, 2, 3], [0, 0, 0]], dtype='int32'
            )
            out_z1 = paddle.index_sample(x, index)

            except_output = np.array(
                [[1.0, 2.0, 3.0], [6.0, 7.0, 8.0], [9.0, 9.0, 9.0]]
            )
            assert out_z1.numpy().all() == except_output.all()


support_types = get_xpu_op_support_types('index_sample')
for stype in support_types:
    create_test_class(globals(), XPUTestIndexSampleOP, stype)

if __name__ == "__main__":
    unittest.main()
