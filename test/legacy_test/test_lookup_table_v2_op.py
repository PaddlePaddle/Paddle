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

import unittest

import numpy as np
from op import Operator
from op_test import OpTest, convert_float_to_uint16, skip_check_grad_ci

import paddle
from paddle import base
from paddle.base import Program, core, program_guard


class TestStaticGraphSupportMultipleInt(unittest.TestCase):
    def test_main(self):
        dtypes = ['uint8', 'int8', 'int16', 'int32', 'int64']
        if paddle.in_dynamic_mode():
            paddle.enable_static()
            disable_static = True
        else:
            disable_static = False
        for i, dtype in enumerate(dtypes):
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x = paddle.static.data(name='x', shape=[-1, 7, 30], dtype=dtype)
                emb = paddle.nn.Embedding(10, 20)
                y = emb(x)

        if disable_static:
            paddle.disable_static()


class TestLookupTableOp(OpTest):
    def setUp(self):
        self.op_type = "lookup_table_v2"
        self.python_api = paddle.nn.functional.embedding
        self.init_dtype()

        table = np.random.random((17, 31)).astype(self.dtype)
        ids = np.random.randint(0, 17, 4).astype(self.id_dtype())

        self.inputs = {'W': table, 'Ids': ids}
        self.outputs = {'Out': table[ids]}

    def init_dtype(self):
        self.dtype = "float64"

    def id_dtype(self):
        return "int64"

    def test_check_output(self):
        self.check_output(check_cinn=True, check_new_ir=True)

    def test_check_grad(self):
        self.check_grad(
            ['W'],
            'Out',
            no_grad_set=set('Ids'),
            check_cinn=True,
            check_new_ir=True,
        )


class TestLookupTableOpInt16(OpTest):
    def id_dtype(self):
        return "int16"


class TestLookupTableOpInt8(OpTest):
    def id_dtype(self):
        return "int8"


class TestLookupTableOpUInt8(OpTest):
    def id_dtype(self):
        return "uint8"


class TestLookupTableOpWithTensorIds(OpTest):
    def setUp(self):
        self.op_type = "lookup_table_v2"
        self.python_api = paddle.nn.functional.embedding
        table = np.random.random((17, 31)).astype("float64")
        ids = np.random.randint(low=0, high=17, size=(2, 4, 5)).astype("int32")
        self.inputs = {'W': table, 'Ids': ids}
        self.outputs = {'Out': table[ids.flatten()].reshape((2, 4, 5, 31))}

    def test_check_output(self):
        self.check_output(check_cinn=True, check_new_ir=True)

    def test_check_grad(self):
        self.check_grad(
            ['W'],
            'Out',
            no_grad_set=set('Ids'),
            check_cinn=True,
            check_new_ir=True,
        )


@skip_check_grad_ci(
    reason="Since paddings are not trainable and fixed in forward,"
    "the gradient of paddings makes no sense and we don't "
    "test the gradient here."
)
class TestLookupTableOpWithPadding(TestLookupTableOp):
    def test_check_output(self):
        ids = np.squeeze(self.inputs['Ids'])
        padding_idx = np.random.choice(ids, 1)[0]
        self.outputs['Out'][ids == padding_idx] = np.zeros(31)
        self.attrs = {'padding_idx': int(padding_idx)}
        self.check_output(check_cinn=True, check_new_ir=True)


@skip_check_grad_ci(
    reason="Since paddings are not trainable and fixed in forward,"
    "the gradient of paddings makes no sense and we don't "
    "test the gradient here."
)
class TestLookupTableOpWithTensorIdsAndPadding(TestLookupTableOpWithTensorIds):
    def test_check_output(self):
        ids = self.inputs['Ids']
        flatten_idx = ids.flatten()
        padding_idx = np.random.choice(flatten_idx, 1)[0]
        self.outputs['Out'][np.squeeze(ids == padding_idx)] = np.zeros(31)
        self.attrs = {'padding_idx': padding_idx}
        self.check_output(check_cinn=True, check_new_ir=True)


class TestLookupTableWIsSelectedRows(unittest.TestCase):
    def prepare_ids(self, scope, place):
        ids_tensor = scope.var('Ids').get_tensor()
        ids_array = np.array([0, 4, 3, 5]).astype("int32")
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
            assert (row == result_array[idx]).all()

    def check_with_place(self, place):
        scope = core.Scope()

        ids_array = self.prepare_ids(scope, place)

        self.prepare_w(scope, place)

        out_tensor = self.create_out_tensor(scope, place)

        # create and run lookup_table operator
        lookup_table = Operator("lookup_table_v2", W='W', Ids='Ids', Out='Out')
        lookup_table.run(scope, place)

        # get result from Out
        result_array = np.array(out_tensor)

        self.check_result(ids_array, result_array)

    def test_w_is_selected_rows(self):
        places = [core.CPUPlace()]
        # currently only support CPU
        for place in places:
            self.check_with_place(place)


class TestLookupTableWithTensorIdsWIsSelectedRows(
    TestLookupTableWIsSelectedRows
):
    def prepare_ids(self, scope, place):
        ids_tensor = scope.var('Ids').get_tensor()
        ids_array = np.random.randint(low=0, high=6, size=(2, 4, 3)).astype(
            "int64"
        )
        ids_tensor.set(ids_array, place)
        return ids_array

    def check_result(self, ids_array, result_array):
        for idx, row in np.ndenumerate(ids_array):
            assert (row == result_array[idx]).all()


class TestLookupTableIsSparse(unittest.TestCase):
    def init_data(self):
        self.x_data = np.array([[1, 3, 0, 4, 7]]).astype("int64")
        self.y_data = np.array([[0.1, 0.3, 0, 0.4, 0.7]]).astype("float32")

    def get_w_grad(self, is_sparse):
        paddle.enable_static()
        self.init_data()
        main_program = base.Program()
        with base.program_guard(main_program, base.Program()):
            x = paddle.static.data(name='x', shape=[-1, 5], dtype='int64')
            y_ = paddle.static.data(name='y_', shape=[-1, 5], dtype='float32')
            emb = paddle.static.nn.embedding(
                input=x,
                size=[10, 16],
                param_attr=base.ParamAttr(
                    name="emb_weight",
                    learning_rate=10,
                    initializer=paddle.nn.initializer.Assign(self.w_data),
                ),
                is_sparse=is_sparse,
            )
            y = paddle.sum(emb, axis=-1)

            loss = paddle.nn.functional.square_error_cost(input=y, label=y_)
            loss = paddle.mean(loss)

            sgd_optimizer = paddle.optimizer.SGD(learning_rate=1e-4)
            sgd_optimizer.minimize(loss)

            place = base.CPUPlace()
            exe = base.Executor(place)
            exe.run(base.default_startup_program())
            ret = exe.run(
                feed={'x': self.x_data, 'y_': self.y_data},
                fetch_list=['emb_weight'],
                return_numpy=False,
            )
            return np.array(ret[0])

    def test_w_grad(self):
        self.w_data = np.random.random(size=(10, 16)).astype("float32")
        w_grad = self.get_w_grad(False)
        w_grad_with_sparse = self.get_w_grad(True)
        self.check_grad(w_grad, w_grad_with_sparse)

    def check_grad(self, w_grad1, w_grad2, tolerance=1e-6):
        np.testing.assert_allclose(
            w_grad1, w_grad2, rtol=tolerance, atol=tolerance
        )


class TestLookupTableApi(unittest.TestCase):
    def test_api(self):
        paddle.enable_static()
        x = paddle.static.data(name='x', shape=[-1, 20], dtype='int64')
        emb = paddle.static.nn.embedding(input=x, size=[128, 64])

        place = base.CPUPlace()
        x_data = np.random.randint(0, 127, [2, 20]).astype("int64")

        exe = base.Executor(place)
        exe.run(base.default_startup_program())
        ret = exe.run(
            feed={
                'x': x_data,
            },
            fetch_list=[emb],
            return_numpy=False,
        )


class TestEmbedOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            input_data = np.random.randint(0, 10, (4, 6)).astype("int64")

            def test_Variable():
                # the input type must be Variable
                paddle.static.nn.embedding(input=input_data, size=(10, 64))

            self.assertRaises(TypeError, test_Variable)

            def test_input_dtype():
                # the input dtype must be int64
                input = paddle.static.data(
                    name='x1', shape=[4, 6], dtype='float32'
                )
                paddle.static.nn.embedding(input=input, size=(10, 64))

            self.assertRaises(TypeError, test_input_dtype)

            def test_param_dtype():
                # dtype must be float32 or float64
                input2 = paddle.static.data(
                    name='x2', shape=[4, 6], dtype='int64'
                )
                paddle.static.nn.embedding(
                    input=input2, size=(10, 64), dtype='int64'
                )

            self.assertRaises(TypeError, test_param_dtype)
            input3 = paddle.static.data(name='x3', shape=[4, 6], dtype='int64')
            paddle.static.nn.embedding(
                input=input3, size=(10, 64), dtype='float16'
            )


class TestEmbeddingFP16OP(TestLookupTableOp):
    def setUp(self):
        self.op_type = "lookup_table_v2"
        self.python_api = paddle.nn.functional.embedding
        self.init_dtype()

        table = np.random.random((18, 32)).astype(self.dtype)
        ids = np.random.randint(0, 18, 4).astype(self.id_dtype())

        self.inputs = {'W': table, 'Ids': ids}
        self.outputs = {'Out': table[ids]}

    def init_dtype(self):
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not complied with CUDA and not support the bfloat16",
)
class TestEmbeddingBF16OP(OpTest):
    def setUp(self):
        self.op_type = "lookup_table_v2"
        self.python_api = paddle.nn.functional.embedding
        self.dtype = np.uint16

        table = np.random.random((18, 32)).astype("float32")
        ids = np.random.randint(0, 18, 4).astype(self.id_dtype())

        self.inputs = {'W': convert_float_to_uint16(table), 'Ids': ids}
        self.outputs = {'Out': convert_float_to_uint16(table[ids])}

    def id_dtype(self):
        return "int64"

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, check_cinn=True, check_new_ir=True)

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place,
            ['W'],
            'Out',
            no_grad_set=set('Ids'),
            check_cinn=True,
            check_new_ir=True,
        )


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
