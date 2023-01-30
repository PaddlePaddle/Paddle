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

<<<<<<< HEAD
import sys
import unittest

import numpy as np

sys.path.append("..")
from op_test import OpTest, skip_check_grad_ci

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import Program, program_guard
from paddle.fluid.op import Operator
=======
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
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

paddle.enable_static()


class TestLookupTableOp(OpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "lookup_table_v2"
        table = np.random.random((17, 31)).astype("float64")
        ids = np.random.randint(0, 17, 4).astype("int64")
        self.inputs = {'W': table, 'Ids': ids}
        self.outputs = {'Out': table[ids]}

    def test_check_output_with_place(self):
        self.check_output_with_place(place=paddle.XPUPlace(0))

    def test_check_grad(self):

<<<<<<< HEAD
        self.check_grad_with_place(
            inputs_to_check=['W'],
            output_names='Out',
            no_grad_set=set('Ids'),
            place=paddle.XPUPlace(0),
            in_place=True,
        )


class TestLookupTableOpWithTensorIds(OpTest):
=======
        self.check_grad_with_place(inputs_to_check=['W'],
                                   output_names='Out',
                                   no_grad_set=set('Ids'),
                                   place=paddle.XPUPlace(0),
                                   in_place=True)


class TestLookupTableOpWithTensorIds(OpTest):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "lookup_table_v2"
        table = np.random.random((17, 31)).astype("float64")
        ids = np.random.randint(low=0, high=17, size=(2, 4, 5)).astype("int32")
        self.inputs = {'W': table, 'Ids': ids}
        self.outputs = {'Out': table[ids.flatten()].reshape((2, 4, 5, 31))}

    def test_check_output(self):
        self.check_output_with_place(place=paddle.XPUPlace(0))

    def test_check_grad(self):
<<<<<<< HEAD
        self.check_grad_with_place(
            inputs_to_check=['W'],
            output_names='Out',
            no_grad_set=set('Ids'),
            place=paddle.XPUPlace(0),
            in_place=True,
        )
=======
        self.check_grad_with_place(inputs_to_check=['W'],
                                   output_names='Out',
                                   no_grad_set=set('Ids'),
                                   place=paddle.XPUPlace(0),
                                   in_place=True)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


@skip_check_grad_ci(
    reason="Since paddings are not trainable and fixed in forward,"
    "the gradient of paddings makes no sense and we don't "
<<<<<<< HEAD
    "test the gradient here."
)
class TestLookupTableOpWithPadding(TestLookupTableOp):
=======
    "test the gradient here.")
class TestLookupTableOpWithPadding(TestLookupTableOp):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_check_output(self):
        ids = np.squeeze(self.inputs['Ids'])
        padding_idx = np.random.choice(ids, 1)[0]
        self.outputs['Out'][ids == padding_idx] = np.zeros(31)
        self.attrs = {'padding_idx': int(padding_idx)}
        self.check_output_with_place(place=paddle.XPUPlace(0))


@skip_check_grad_ci(
    reason="Since paddings are not trainable and fixed in forward,"
    "the gradient of paddings makes no sense and we don't "
<<<<<<< HEAD
    "test the gradient here."
)
class TestLookupTableOpWithTensorIdsAndPadding(TestLookupTableOpWithTensorIds):
=======
    "test the gradient here.")
class TestLookupTableOpWithTensorIdsAndPadding(TestLookupTableOpWithTensorIds):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_check_output(self):
        ids = self.inputs['Ids']
        flatten_idx = ids.flatten()
        padding_idx = np.random.choice(flatten_idx, 1)[0]
        self.outputs['Out'][np.squeeze(ids == padding_idx)] = np.zeros(31)
<<<<<<< HEAD
        self.attrs = {'padding_idx': padding_idx}
=======
        self.attrs = {'padding_idx': cpt.long_type(padding_idx)}
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.check_output_with_place(place=paddle.XPUPlace(0))


class TestLookupTableWIsSelectedRows(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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


<<<<<<< HEAD
class TestLookupTableWithTensorIdsWIsSelectedRows(
    TestLookupTableWIsSelectedRows
):
    def prepare_ids(self, scope, place):
        ids_tensor = scope.var('Ids').get_tensor()
        ids_array = np.random.randint(low=0, high=6, size=(2, 4, 3)).astype(
            "int64"
        )
=======
class TestLookupTableWithTensorIdsWIsSelectedRows(TestLookupTableWIsSelectedRows
                                                  ):

    def prepare_ids(self, scope, place):
        ids_tensor = scope.var('Ids').get_tensor()
        ids_array = np.random.randint(low=0, high=6,
                                      size=(2, 4, 3)).astype("int64")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        ids_tensor.set(ids_array, place)
        return ids_array

    def check_result(self, ids_array, result_array):
        for idx, row in np.ndenumerate(ids_array):
            assert (row == result_array[idx]).all()


class TestLookupTableApi(unittest.TestCase):
<<<<<<< HEAD
    def test_api(self):
        x = paddle.static.data(name='x', shape=[-1, 20], dtype='int64')
=======

    def test_api(self):
        x = fluid.layers.data(name='x', shape=[20], dtype='int64')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        emb = fluid.embedding(input=x, size=[128, 64])

        place = paddle.XPUPlace(0)
        x_data = np.random.randint(0, 127, [2, 20]).astype("int64")

        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
<<<<<<< HEAD
        ret = exe.run(
            feed={
                'x': x_data,
            },
            fetch_list=[emb],
            return_numpy=False,
        )


class TestEmbedOpError(unittest.TestCase):
=======
        ret = exe.run(feed={
            'x': x_data,
        },
                      fetch_list=[emb],
                      return_numpy=False)


class TestEmbedOpError(unittest.TestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
