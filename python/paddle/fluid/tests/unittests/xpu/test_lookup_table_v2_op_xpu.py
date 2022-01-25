#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np
import sys
sys.path.append("..")
from op_test import OpTest, skip_check_grad_ci
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid.op import Operator
import paddle.compat as cpt
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard

import op_test
from op_test import OpTest, skip_check_grad_ci
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper

paddle.enable_static()


class TestDygraphEmbeddingAPIError(unittest.TestCase):
    def test_errors(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            dict_size = 20
            layer = fluid.dygraph.nn.Embedding(
                size=[dict_size, 32], param_attr='emb.w', is_sparse=False)
            # the input must be Variable
            x0 = fluid.create_lod_tensor(
                np.array([-1, 3, 5, 5]), [[1, 1, 1, 1]], paddle.XPUPlace(0))
            self.assertRaises(TypeError, layer, x0)
            # the input dtype must be int64
            data_t = fluid.data(name='word', shape=[1], dtype='int32')
            self.assertRaises(TypeError, layer, data_t)


class TestLookupTableOp(OpTest):
    def setUp(self):
        self.op_type = "lookup_table_v2"
        table = np.random.random((17, 31)).astype("float64")
        ids = np.random.randint(0, 17, 4).astype("int64")
        self.inputs = {'W': table, 'Ids': ids}
        self.outputs = {'Out': table[ids]}

    def test_check_output_with_place(self):
        self.check_output_with_place(place=paddle.XPUPlace(0))

    def test_check_grad(self):

        self.check_grad_with_place(
            inputs_to_check=['W'],
            output_names='Out',
            no_grad_set=set('Ids'),
            place=paddle.XPUPlace(0),
            in_place=True)


class TestLookupTableOpWithTensorIds(OpTest):
    def setUp(self):
        self.op_type = "lookup_table_v2"
        table = np.random.random((17, 31)).astype("float64")
        ids = np.random.randint(low=0, high=17, size=(2, 4, 5)).astype("int32")
        self.inputs = {'W': table, 'Ids': ids}
        self.outputs = {'Out': table[ids.flatten()].reshape((2, 4, 5, 31))}

    def test_check_output(self):
        self.check_output_with_place(place=paddle.XPUPlace(0))

    def test_check_grad(self):
        self.check_grad_with_place(
            inputs_to_check=['W'],
            output_names='Out',
            no_grad_set=set('Ids'),
            place=paddle.XPUPlace(0),
            in_place=True)


@skip_check_grad_ci(
    reason="Since paddings are not trainable and fixed in forward,"
    "the gradient of paddings makes no sense and we don't "
    "test the gradient here.")
class TestLookupTableOpWithPadding(TestLookupTableOp):
    def test_check_output(self):
        ids = np.squeeze(self.inputs['Ids'])
        padding_idx = np.random.choice(ids, 1)[0]
        self.outputs['Out'][ids == padding_idx] = np.zeros(31)
        self.attrs = {'padding_idx': int(padding_idx)}
        self.check_output_with_place(place=paddle.XPUPlace(0))


@skip_check_grad_ci(
    reason="Since paddings are not trainable and fixed in forward,"
    "the gradient of paddings makes no sense and we don't "
    "test the gradient here.")
class TestLookupTableOpWithTensorIdsAndPadding(TestLookupTableOpWithTensorIds):
    def test_check_output(self):
        ids = self.inputs['Ids']
        flatten_idx = ids.flatten()
        padding_idx = np.random.choice(flatten_idx, 1)[0]
        self.outputs['Out'][np.squeeze(ids == padding_idx)] = np.zeros(31)
        self.attrs = {'padding_idx': cpt.long_type(padding_idx)}
        self.check_output_with_place(place=paddle.XPUPlace(0))


class TestLookupTableWIsSelectedRows(unittest.TestCase):
    def prepare_ids(self, scope, place):
        ids_tensor = scope.var('Ids').get_tensor()
        ids_array = np.array([0, 4, 3, 5]).astype("int64")
        ids_tensor.set(ids_array, place)
        return ids_array

    def prepare_w(self, scope, place):
        rows = [0, 1, 2, 3, 4, 5, 6]
        row_numel = 12
        w_selected_rows = scope.var('W')
        w_array = np.ones((len(rows), row_numel)).astype("float32")
        for i in range(len(rows)):
            w_array[i] *= i
        w_tensor = w_selected_rows.get_tensor()
        w_tensor.set(w_array, place)

    def create_out_tensor(self, scope, place):
        return scope.var('Out').get_tensor()

    def check_result(self, ids_array, result_array):
        # all(): return True if all elements of the iterable are true (or if the iterable is empty)
        for idx, row in enumerate(ids_array):
            assert (row == result_array[idx]).all()

    def check_with_place(self, place):
        scope = core.Scope()
        ids_array = self.prepare_ids(scope, place)

        self.prepare_w(scope, place)

        out_tensor = self.create_out_tensor(scope, place)

        # create and run lookup_table_v2 operator
        lookup_table = Operator("lookup_table_v2", W='W', Ids='Ids', Out='Out')
        lookup_table.run(scope, place)

        # get result from Out
        result_array = np.array(out_tensor)

        self.check_result(ids_array, result_array)

    def test_w_is_selected_rows(self):
        places = [paddle.XPUPlace(0)]
        for place in places:
            self.check_with_place(place)


class TestLookupTableWithTensorIdsWIsSelectedRows(
        TestLookupTableWIsSelectedRows):
    def prepare_ids(self, scope, place):
        ids_tensor = scope.var('Ids').get_tensor()
        ids_array = np.random.randint(
            low=0, high=6, size=(2, 4, 3)).astype("int64")
        ids_tensor.set(ids_array, place)
        return ids_array

    def check_result(self, ids_array, result_array):
        for idx, row in np.ndenumerate(ids_array):
            assert (row == result_array[idx]).all()


class XPUTestLookupTableIsSparse(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'lookup_table_v2_grad'
        self.use_dynamic_create_class = False

    class TestLookupTableIsSparse(XPUOpTest):
        def setUp(self):
            self.set_xpu()
            self.op_type = 'lookup_table_v2_grad'
            self.place = paddle.XPUPlace(0)

            self.init_dtype()
            self.set_inputs()

        def set_inputs(self):
            shape, _ = self.set_shape()
            self.x_data = np.random.randint(10, size=shape).astype("int64")
            self.y_data = np.random.uniform(0, 1., shape).astype(self.dtype)

        def set_w_grad(self, is_sparse):
            self.set_inputs()
            main_program = fluid.Program()
            data_shape, emb_shape = self.set_shape()
            with fluid.program_guard(main_program, fluid.Program()):
                x = fluid.layers.data(
                    name='x', shape=[data_shape], dtype='int64')
                y_ = fluid.layers.data(
                    name='y_', shape=[data_shape], dtype=self.dtype)
                emb = fluid.input.embedding(
                    input=x,
                    size=emb_shape,
                    param_attr=fluid.ParamAttr(
                        name="emb_weight",
                        learning_rate=10,
                        initializer=fluid.initializer.NumpyArrayInitializer(
                            self.w_data)),
                    is_sparse=is_sparse)
                y = fluid.layers.reduce_sum(emb, dim=-1)

                loss = fluid.layers.square_error_cost(input=y, label=y_)
                loss = fluid.layers.mean(loss)

                adam_optimizer = paddle.optimizer.Adam(learning_rate=1e-3)
                adam_optimizer.minimize(loss)

                place = paddle.XPUPlace(0)
                exe = fluid.Executor(place)
                exe.run(fluid.default_startup_program())
                ret = exe.run(feed={'x': self.x_data,
                                    'y_': self.y_data},
                              fetch_list=['emb_weight'],
                              return_numpy=False)
                return np.array(ret[0])

        def check_grad(self, w_grad1, w_grad2, tolerance=1e-6):
            np.testing.assert_allclose(
                w_grad1, w_grad2, rtol=tolerance, atol=tolerance)

        def test_check_w_grad(self):
            _, shape = self.set_shape()
            self.w_data = np.random.random(size=shape).astype(self.dtype)
            w_grad = self.set_w_grad(False)
            w_grad_with_sparse = self.set_w_grad(True)
            self.check_grad(w_grad, w_grad_with_sparse)

        def set_shape(self):
            return (5, (10, 7))

        def set_xpu(self):
            self.__class__.use_xpu = True

        def init_dtype(self):
            self.dtype = self.in_type

    class TestLookupTableIsSparse1(TestLookupTableIsSparse):
        def set_shape(self):
            return (6, 64)

    class TestLookupTableIsSparse2(TestLookupTableIsSparse):
        def set_shape(self):
            return (7, (10, 10))

    class TestLookupTableIsSparse3(TestLookupTableIsSparse):
        def set_shape(self):
            return ((13, 9), (52, 9))

    class TestLookupTableIsSparse4(TestLookupTableIsSparse):
        def set_shape(self):
            return ((47, 1), (10, 10, 1))

    class TestLookupTableIsSparse5(TestLookupTableIsSparse):
        def set_shape(self):
            return (10, (512, 127, 1))


class TestLookupTableApi(unittest.TestCase):
    def test_api(self):
        x = fluid.layers.data(name='x', shape=[20], dtype='int64')
        emb = fluid.embedding(input=x, size=[128, 64])

        place = paddle.XPUPlace(0)
        x_data = np.random.randint(0, 127, [2, 20]).astype("int64")

        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        ret = exe.run(feed={'x': x_data, },
                      fetch_list=[emb],
                      return_numpy=False)


class TestEmbedOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            input_data = np.random.randint(0, 10, (4, 6)).astype("int64")

            def test_Variable():
                # the input type must be Variable
                fluid.embedding(input=input_data, size=(10, 64))

            self.assertRaises(TypeError, test_Variable)

            def test_input_dtype():
                # the input dtype must be int64
                input = fluid.data(name='x1', shape=[4, 6], dtype='float32')
                fluid.embedding(input=input, size=(10, 64))

            self.assertRaises(TypeError, test_input_dtype)

            def test_param_dtype():
                # dtype must be float32 or float64
                input2 = fluid.data(name='x2', shape=[4, 6], dtype='int64')
                fluid.embedding(input=input2, size=(10, 64), dtype='int64')

            self.assertRaises(TypeError, test_param_dtype)
            input3 = fluid.data(name='x3', shape=[4, 6], dtype='int64')
            fluid.embedding(input=input3, size=(10, 64), dtype='float16')


support_types = get_xpu_op_support_types('lookup_table_v2_grad')
for stype in support_types:
    create_test_class(
        globals(),
        XPUTestLookupTableIsSparse,
        stype,
        ignore_deivce_version=[core.XPUVersion.XPU1])

if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
