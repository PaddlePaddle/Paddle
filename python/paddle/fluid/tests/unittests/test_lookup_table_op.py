#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest, skip_check_grad_ci, check_out_dtype
import paddle.fluid.core as core
from paddle.fluid.op import Operator
import paddle.compat as cpt
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard
import paddle.nn.functional as F


class TestLookupTableOp(OpTest):

    def setUp(self):
        self.op_type = "lookup_table"
        table = np.random.random((17, 31)).astype("float64")
        ids = np.random.randint(0, 17, 4).astype("int64")
        ids_expand = np.expand_dims(ids, axis=1)
        self.inputs = {'W': table, 'Ids': ids_expand}
        self.outputs = {'Out': table[ids]}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['W'], 'Out', no_grad_set=set('Ids'))


class TestLookupTableOpWithTensorIds(OpTest):

    def setUp(self):
        self.op_type = "lookup_table"
        table = np.random.random((17, 31)).astype("float64")
        ids = np.random.randint(low=0, high=17,
                                size=(2, 4, 5, 1)).astype("int64")
        self.inputs = {'W': table, 'Ids': ids}
        self.outputs = {'Out': table[ids.flatten()].reshape((2, 4, 5, 31))}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['W'], 'Out', no_grad_set=set('Ids'))


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
        self.check_output()


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
        self.check_output()


class TestLookupTableWIsSelectedRows(unittest.TestCase):

    def prepare_ids(self, scope, place):
        ids_tensor = scope.var('Ids').get_tensor()
        ids_array = np.array([[0], [4], [3], [5]]).astype("int64")
        ids_tensor.set(ids_array, place)
        return ids_array

    def prepare_w(self, scope, place):
        rows = [0, 1, 2, 3, 4, 5, 6]
        row_numel = 12

        w_selected_rows = scope.var('W').get_selected_rows()
        w_selected_rows.set_height(len(rows))
        w_selected_rows.set_rows(rows)
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
            assert (row[0] == result_array[idx]).all()

    def check_with_place(self, place):
        scope = core.Scope()

        ids_array = self.prepare_ids(scope, place)

        self.prepare_w(scope, place)

        out_tensor = self.create_out_tensor(scope, place)

        # create and run lookup_table operator
        lookup_table = Operator("lookup_table", W='W', Ids='Ids', Out='Out')
        lookup_table.run(scope, place)

        # get result from Out
        result_array = np.array(out_tensor)

        self.check_result(ids_array, result_array)

    def test_w_is_selected_rows(self):
        places = [core.CPUPlace()]
        # currently only support CPU
        for place in places:
            self.check_with_place(place)


class TestLookupTableWithTensorIdsWIsSelectedRows(TestLookupTableWIsSelectedRows
                                                  ):

    def prepare_ids(self, scope, place):
        ids_tensor = scope.var('Ids').get_tensor()
        ids_array = np.random.randint(low=0, high=6,
                                      size=(2, 4, 3, 1)).astype("int64")
        ids_tensor.set(ids_array, place)
        return ids_array

    def check_result(self, ids_array, result_array):
        for idx, row in np.ndenumerate(ids_array):
            assert (row == result_array[idx]).all()


class TestEmbedOpError(unittest.TestCase):

    def test_errors(self):
        with program_guard(Program(), Program()):
            input_data = np.random.randint(0, 10, (4, 1)).astype("int64")

            def test_Variable():
                # the input type must be Variable
                fluid.layers.embedding(input=input_data, size=(10, 64))

            self.assertRaises(TypeError, test_Variable)

            def test_input_dtype():
                # the input dtype must be int64
                input = fluid.data(name='x', shape=[4, 1], dtype='float32')
                fluid.layers.embedding(input=input, size=(10, 64))

            self.assertRaises(TypeError, test_input_dtype)

            def test_param_dtype():
                # dtype must be float32 or float64
                input2 = fluid.data(name='x2', shape=[4, 1], dtype='int64')
                fluid.layers.embedding(input=input2,
                                       size=(10, 64),
                                       dtype='int64')

            self.assertRaises(TypeError, test_param_dtype)

            input3 = fluid.data(name='x3', shape=[4, 1], dtype='int64')
            fluid.layers.embedding(input=input3, size=(10, 64), dtype='float16')


class TestLookupTableOpInt8(OpTest):

    def setUp(self):
        self.op_type = "lookup_table"
        table = np.random.randint(low=-128, high=127,
                                  size=(17, 31)).astype("int8")
        ids = np.random.randint(0, 17, 4).astype("int64")
        ids_expand = np.expand_dims(ids, axis=1)
        self.inputs = {'W': table, 'Ids': ids_expand}
        self.outputs = {'Out': table[ids]}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        # since int8 type only be used in test and inference, there is
        # no gradient implement, so we don't need to test it
        pass


class TestLookupTableOpWithTensorIdsInt8(OpTest):

    def setUp(self):
        self.op_type = "lookup_table"
        table = np.random.randint(low=-128, high=127,
                                  size=(17, 31)).astype("int8")
        ids = np.random.randint(low=0, high=17,
                                size=(2, 4, 5, 1)).astype("int64")
        self.inputs = {'W': table, 'Ids': ids}
        self.outputs = {'Out': table[ids.flatten()].reshape((2, 4, 5, 31))}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        # since int8 type only be used in test and inference, there is
        # no gradient implement, so we don't need to test it
        pass


class TestLookupTableOpWithPaddingInt8(TestLookupTableOpInt8):

    def test_check_output(self):
        ids = np.squeeze(self.inputs['Ids'])
        padding_idx = np.random.choice(ids, 1)[0]
        self.outputs['Out'][ids == padding_idx] = np.zeros(31)
        self.attrs = {'padding_idx': int(padding_idx)}
        self.check_output()

    def test_check_grad(self):
        # Since paddings are not trainable and fixed in forward, the gradient of
        # paddings makes no sense and we don't test the gradient here.
        pass


class TestLookupTableOpWithTensorIdsAndPaddingInt8(
        TestLookupTableOpWithTensorIdsInt8):

    def test_check_output(self):
        ids = self.inputs['Ids']
        flatten_idx = ids.flatten()
        padding_idx = np.random.choice(flatten_idx, 1)[0]
        self.outputs['Out'][np.squeeze(ids == padding_idx)] = np.zeros(31)
        self.attrs = {'padding_idx': cpt.long_type(padding_idx)}
        self.check_output()

    def test_check_grad(self):
        # Since paddings are not trainable and fixed in forward, the gradient of
        # paddings makes no sense and we don't test the gradient here.
        pass


class TestLookupTableWIsSelectedRowsInt8(unittest.TestCase):

    def prepare_ids(self, scope, place):
        ids_tensor = scope.var('Ids').get_tensor()
        ids_array = np.array([[0], [4], [3], [5]]).astype("int64")
        ids_tensor.set(ids_array, place)
        return ids_array

    def prepare_w(self, scope, place):
        rows = [0, 1, 2, 3, 4, 5, 6]
        row_numel = 12

        w_selected_rows = scope.var('W').get_selected_rows()
        w_selected_rows.set_height(len(rows))
        w_selected_rows.set_rows(rows)
        w_array = np.ones((len(rows), row_numel)).astype("int8")
        for i in range(len(rows)):
            w_array[i] *= i
        w_tensor = w_selected_rows.get_tensor()
        w_tensor.set(w_array, place)

    def create_out_tensor(self, scope, place):
        return scope.var('Out').get_tensor()

    def check_result(self, ids_array, result_array):
        # all(): return True if all elements of the iterable are true (or if the iterable is empty)
        for idx, row in enumerate(ids_array):
            assert (row[0] == result_array[idx]).all()

    def check_with_place(self, place):
        scope = core.Scope()

        ids_array = self.prepare_ids(scope, place)

        self.prepare_w(scope, place)

        out_tensor = self.create_out_tensor(scope, place)

        # create and run lookup_table operator
        lookup_table = Operator("lookup_table", W='W', Ids='Ids', Out='Out')
        lookup_table.run(scope, place)

        # get result from Out
        result_array = np.array(out_tensor)

        self.check_result(ids_array, result_array)

    def test_w_is_selected_rows(self):
        places = [core.CPUPlace()]
        # currently only support CPU
        for place in places:
            self.check_with_place(place)


class TestLookupTableWithTensorIdsWIsSelectedRowsInt8(
        TestLookupTableWIsSelectedRowsInt8):

    def prepare_ids(self, scope, place):
        ids_tensor = scope.var('Ids').get_tensor()
        ids_array = np.random.randint(low=0, high=6,
                                      size=(2, 4, 3, 1)).astype("int64")
        ids_tensor.set(ids_array, place)
        return ids_array

    def check_result(self, ids_array, result_array):
        for idx, row in np.ndenumerate(ids_array):
            assert (row == result_array[idx]).all()


@skip_check_grad_ci(reason="Int16 type only be used in test and inference.")
class TestLookupTableOpInt16(OpTest):

    def setUp(self):
        self.op_type = "lookup_table"
        table = np.random.randint(low=-128, high=127,
                                  size=(17, 31)).astype("int16")
        ids = np.random.randint(0, 17, 4).astype("int64")
        ids_expand = np.expand_dims(ids, axis=1)
        self.inputs = {'W': table, 'Ids': ids_expand}
        self.outputs = {'Out': table[ids]}

    def test_check_output(self):
        self.check_output()


@skip_check_grad_ci(reason="Int16 type only be used in test and inference.")
class TestLookupTableOpWithTensorIdsInt16(OpTest):

    def setUp(self):
        self.op_type = "lookup_table"
        table = np.random.randint(low=-128, high=127,
                                  size=(17, 31)).astype("int16")
        ids = np.random.randint(low=0, high=17,
                                size=(2, 4, 5, 1)).astype("int64")
        self.inputs = {'W': table, 'Ids': ids}
        self.outputs = {'Out': table[ids.flatten()].reshape((2, 4, 5, 31))}

    def test_check_output(self):
        self.check_output()


@skip_check_grad_ci(reason="Int16 type only be used in test and inference.")
class TestLookupTableOpWithPaddingInt16(TestLookupTableOpInt16):

    def test_check_output(self):
        ids = np.squeeze(self.inputs['Ids'])
        padding_idx = np.random.choice(ids, 1)[0]
        self.outputs['Out'][ids == padding_idx] = np.zeros(31)
        self.attrs = {'padding_idx': int(padding_idx)}
        self.check_output()


@skip_check_grad_ci(reason="Int16 type only be used in test and inference.")
class TestLookupTableOpWithTensorIdsAndPaddingInt16(
        TestLookupTableOpWithTensorIdsInt16):

    def test_check_output(self):
        ids = self.inputs['Ids']
        flatten_idx = ids.flatten()
        padding_idx = np.random.choice(flatten_idx, 1)[0]
        self.outputs['Out'][np.squeeze(ids == padding_idx)] = np.zeros(31)
        self.attrs = {'padding_idx': cpt.long_type(padding_idx)}
        self.check_output()


class TestLookupTableWIsSelectedRowsInt16(unittest.TestCase):

    def prepare_ids(self, scope, place):
        ids_tensor = scope.var('Ids').get_tensor()
        ids_array = np.array([[0], [4], [3], [5]]).astype("int64")
        ids_tensor.set(ids_array, place)
        return ids_array

    def prepare_w(self, scope, place):
        rows = [0, 1, 2, 3, 4, 5, 6]
        row_numel = 12

        w_selected_rows = scope.var('W').get_selected_rows()
        w_selected_rows.set_height(len(rows))
        w_selected_rows.set_rows(rows)
        w_array = np.ones((len(rows), row_numel)).astype("int16")
        for i in range(len(rows)):
            w_array[i] *= i
        w_tensor = w_selected_rows.get_tensor()
        w_tensor.set(w_array, place)

    def create_out_tensor(self, scope, place):
        return scope.var('Out').get_tensor()

    def check_result(self, ids_array, result_array):
        for idx, row in enumerate(ids_array):
            assert (row[0] == result_array[idx]).all()

    def check_with_place(self, place):
        scope = core.Scope()

        ids_array = self.prepare_ids(scope, place)

        self.prepare_w(scope, place)

        out_tensor = self.create_out_tensor(scope, place)

        # create and run lookup_table operator
        lookup_table = Operator("lookup_table", W='W', Ids='Ids', Out='Out')
        lookup_table.run(scope, place)

        # get result from Out
        result_array = np.array(out_tensor)

        self.check_result(ids_array, result_array)

    def test_w_is_selected_rows(self):
        places = [core.CPUPlace()]
        # currently only support CPU
        for place in places:
            self.check_with_place(place)


class TestLookupTableWithTensorIdsWIsSelectedRowsInt16(
        TestLookupTableWIsSelectedRowsInt16):

    def prepare_ids(self, scope, place):
        ids_tensor = scope.var('Ids').get_tensor()
        ids_array = np.random.randint(low=0, high=6,
                                      size=(2, 4, 3, 1)).astype("int64")
        ids_tensor.set(ids_array, place)
        return ids_array

    def check_result(self, ids_array, result_array):
        for idx, row in np.ndenumerate(ids_array):
            assert (row == result_array[idx]).all()


class TestOutDtype(unittest.TestCase):

    def test_dtype(self):
        api_fn = F.embedding
        check_out_dtype(api_fn,
                        in_specs=[([10, 16], 'int64'), ([100, 64], )],
                        expect_dtypes=['float32', 'float64'],
                        target_index=1)


if __name__ == "__main__":
    unittest.main()
