#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import struct
import unittest

import numpy as np

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.static.amp as amp
from paddle.fluid.op import Operator
from paddle.fluid.tests.unittests.op_test import (
    OpTest,
    OpTestTool,
    convert_float_to_uint16,
    convert_uint16_to_float,
)


@unittest.skipIf(
    not core.supports_bfloat16(), 'place does not support BF16 evaluation'
)
class TestSGDOpBF16(OpTest):
    def setUp(self):
        self.op_type = 'sgd'
        self.dtype = np.uint16
        self.use_mkldnn = True
        self.conf()
        w = np.random.random((self.h, self.w)).astype('float32')
        w_bf16 = convert_float_to_uint16(w)
        g = np.random.random((self.h, self.w)).astype('float32')
        g_bf16 = convert_float_to_uint16(g)
        lr = np.array([0.1]).astype('float32')
        lr_bf16 = convert_float_to_uint16(lr)

        self.inputs = {'Param': w_bf16, 'Grad': g_bf16, 'LearningRate': lr_bf16}
        self.outputs = {'ParamOut': w - lr * g}
        self.attrs = {'use_mkldnn': self.use_mkldnn}

    def conf(self):
        self.h = 102
        self.w = 105

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace(), check_dygraph=False)


@unittest.skipIf(
    not core.supports_bfloat16(), 'place does not support BF16 evaluation'
)
class TestSGDOpBF16Case2(TestSGDOpBF16):
    def conf(self):
        self.h = 10
        self.w = 64


class TestSparseSGDOpBF16(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(12345)

    def ref_optimize(self, params, grad_rows, grad_array, lr_value):
        reference = np.copy(params)
        for index, id in enumerate(grad_rows):
            reference[id] = params[id] - lr_value * grad_array[index]
        return reference

    def check_output(self, actual_bf16, reference, atol=0, rtol=0.15e-2):
        actual_fp32 = convert_uint16_to_float(actual_bf16)
        np.testing.assert_allclose(actual_fp32, reference, atol=atol, rtol=rtol)

    def create_sparse_grad_var(self, scope, place, height, rows, row_numel):
        grad_selected_rows = scope.var('Grad').get_selected_rows()
        grad_selected_rows.set_height(height)
        grad_selected_rows.set_rows(rows)
        grad_array = np.random.random((len(rows), row_numel)).astype('float32')
        np_array_bf16 = convert_float_to_uint16(grad_array)

        grad_tensor = grad_selected_rows.get_tensor()
        grad_tensor.set(np_array_bf16, place)

        return grad_tensor, grad_array

    def create_dense_param_var(self, scope, place, height, width):
        param_tensor = scope.var('Param').get_tensor()
        param_array = np.random.random((height, width)).astype('float32')
        param_array_bf16 = convert_float_to_uint16(param_array)
        param_tensor.set(param_array_bf16, place)

        return param_tensor, param_array

    def create_sparse_param_var(self, scope, place, height, rows, row_numel):
        param_selected_rows = scope.var('Param').get_selected_rows()
        param_selected_rows.set_height(height)
        param_selected_rows.set_rows(rows)
        param_selected_rows.sync_index()
        param_array = np.random.random((len(rows), row_numel)).astype('float32')
        np_array_bf16 = convert_float_to_uint16(param_array)

        param_tensor = param_selected_rows.get_tensor()
        param_tensor.set(np_array_bf16, place)

        return param_tensor, param_array

    def create_dense_lr_var(self, scope, place):
        lr_tensor = scope.var('LearningRate').get_tensor()
        lr_value = np.random.uniform()
        lr_array = np.full((1), lr_value, np.float32)
        lr_array_bf16 = convert_float_to_uint16(lr_array)
        lr_tensor.set(lr_array_bf16, place)

        return lr_tensor, lr_value


@unittest.skipIf(
    not core.supports_bfloat16(), 'place does not support BF16 evaluation'
)
class TestSparseGradSGDOpBF16(TestSparseSGDOpBF16):
    def setUp(self):
        self.setup_params()

    def setup_params(self):
        self.grad_height = 10
        self.grad_rows = [0, 4, 7]
        self.grad_row_numel = 12

    def test_sparse_grad_sgd(self):
        scope = core.Scope()
        place = core.CPUPlace()
        _, grad_array = self.create_sparse_grad_var(
            scope, place, self.grad_height, self.grad_rows, self.grad_row_numel
        )
        param_tensor, param_array = self.create_dense_param_var(
            scope, place, self.grad_height, self.grad_row_numel
        )
        _, lr_value = self.create_dense_lr_var(scope, place)

        sgd_op = Operator(
            'sgd',
            Param='Param',
            Grad='Grad',
            ParamOut='Param',
            LearningRate='LearningRate',
            use_mkldnn=True,
        )
        sgd_op.run(scope, place)

        reference = self.ref_optimize(
            param_array, self.grad_rows, grad_array, lr_value
        )
        output = np.array(param_tensor)
        self.check_output(output, reference, atol=5e-3, rtol=1e-1)


@unittest.skipIf(
    not core.supports_bfloat16(), 'place does not support BF16 evaluation'
)
class TestSparseGradSGDOpBF16Case2(TestSparseGradSGDOpBF16):
    def setup_params(self):
        self.grad_height = 14
        self.grad_rows = [1, 4, 12, 7, 8]
        self.grad_row_numel = 16


class TestSparseGradSGDOpBF16Case3(TestSparseGradSGDOpBF16):
    def setup_params(self):
        self.grad_height = 10
        self.grad_rows = [0, 4, 7]
        self.grad_row_numel = 120


@unittest.skipIf(
    not core.supports_bfloat16(), 'place does not support BF16 evaluation'
)
class TestSparseGradParamSGDOpBF16(TestSparseSGDOpBF16):
    def setUp(self):
        self.setup_params()

    def setup_params(self):
        self.grad_height = 10
        self.grad_rows = [0, 4, 7]
        self.grad_row_numel = 12
        self.param_rows = [a for a in range(self.grad_height)]

    def test_sparse_param_grad_sgd(self):
        scope = core.Scope()
        place = core.CPUPlace()
        _, grad_array = self.create_sparse_grad_var(
            scope, place, self.grad_height, self.grad_rows, self.grad_row_numel
        )
        param_tensor, param_array = self.create_sparse_param_var(
            scope, place, self.grad_height, self.param_rows, self.grad_row_numel
        )
        _, lr_value = self.create_dense_lr_var(scope, place)

        sgd_op = Operator(
            'sgd',
            Param='Param',
            Grad='Grad',
            ParamOut='Param',
            LearningRate='LearningRate',
            use_mkldnn=True,
        )
        sgd_op.run(scope, place)

        reference = self.ref_optimize(
            param_array, self.grad_rows, grad_array, lr_value
        )
        output = np.array(param_tensor)
        self.check_output(output, reference, atol=5e-3, rtol=1e-1)


class TestSparseGradParamSGDOpBF16Case2(TestSparseGradParamSGDOpBF16):
    def setup_params(self):
        self.grad_height = 14
        self.grad_rows = [1, 4, 12, 7, 8]
        self.grad_row_numel = 16
        self.param_rows = [a for a in range(self.grad_height)]


@OpTestTool.skip_if_not_cpu_bf16()
class TestSGDOpBF16API(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(12345)
        fluid.set_flags({'FLAGS_use_mkldnn': True})

    def setUp(self):
        self.sample_count = 20
        self.value = np.random.random()

        self.ids_shape = (32, 1)
        self.w_shape = (64, 16)
        self.y_shape = (32, 16)
        self.learning_rate = 0.1

        self._set_initializer()

    def _fp322bf16(self, val: np.float32):
        return np.uint16(struct.unpack('<I', struct.pack('<f', val))[0] >> 16)

    def _bf162fp32(self, val: np.uint16):
        return np.float32(struct.unpack('<f', struct.pack('<I', val << 16))[0])

    def _add_bf16(self, lhs: np.uint16, rhs: np.uint16):
        return self._fp322bf16(self._bf162fp32(lhs) + self._bf162fp32(rhs))

    def _sub_bf16(self, lhs: np.uint16, rhs: np.uint16):
        return self._fp322bf16(self._bf162fp32(lhs) - self._bf162fp32(rhs))

    def _mul_bf16(self, lhs: np.uint16, rhs: np.uint16):
        return self._fp322bf16(self._bf162fp32(lhs) * self._bf162fp32(rhs))

    def _reference(self, data, emb_weight, bf16=False):
        emb_out_shape = np.array(
            [self.ids_shape[0], self.w_shape[1]], dtype=np.int64
        )
        mean_grad_value = np.float32(1.0) / np.prod(
            emb_out_shape, dtype=np.float32
        )
        if bf16:
            mean_grad = np.full(
                emb_out_shape, self._fp322bf16(mean_grad_value), dtype=np.uint16
            )
        else:
            mean_grad = np.full(
                emb_out_shape, mean_grad_value, dtype=np.float32
            )
        # add_grad = 1 * mean_grad
        out_dtype = np.uint16 if bf16 else np.float32
        lookup_table_grad = np.zeros(self.w_shape, dtype=out_dtype)

        # indexes may dupplicate
        if bf16:
            for i, idx in enumerate(data):
                idxv = idx[0]
                for j in range(self.w_shape[1]):
                    lookup_table_grad[idxv, j] = self._add_bf16(
                        lookup_table_grad[idxv, j], mean_grad[i, j]
                    )

            ref_grad = np.ndarray(shape=emb_weight.shape, dtype=np.uint16)
            lr_bf16 = self._fp322bf16(self.learning_rate)

            for i, row in enumerate(emb_weight):
                for j, val in enumerate(row):
                    ref_grad[i, j] = self._sub_bf16(
                        val, self._mul_bf16(lr_bf16, lookup_table_grad[i, j])
                    )
        else:
            for i, idx in enumerate(data):
                lookup_table_grad[idx, :] += mean_grad[i]
            ref_grad = emb_weight - self.learning_rate * lookup_table_grad
        return ref_grad

    def _check_output(
        self, actual, reference, bf16=False, atol=0, rtol=0.15e-2
    ):
        output = actual if bf16 else convert_uint16_to_float(actual)
        if bf16:
            np.testing.assert_allclose(output, reference, atol=atol, rtol=rtol)
        else:
            try:
                print('Compare with FP32 values:')
                np.testing.assert_allclose(
                    output, reference, atol=atol, rtol=rtol
                )
            except AssertionError as e:
                print(e)

    def _set_initializer(self):
        self.initializer = fluid.initializer.Constant(value=self.value)

    def _data_reader(self):
        for sample in range(self.sample_count):
            label = -1 * np.random.random(self.y_shape).astype('float32')
            data = np.random.randint(0, 9, self.ids_shape).astype("int64")
            yield data, label

    def test_sgd(self):
        place = fluid.CPUPlace()
        main = fluid.Program()
        with fluid.program_guard(main):
            x = fluid.layers.data(name='X', shape=self.ids_shape, dtype='int64')
            label = fluid.layers.data(
                name='Y', shape=self.y_shape, dtype='uint16'
            )
            emb = fluid.layers.embedding(
                input=x,
                size=self.w_shape,
                param_attr=fluid.ParamAttr(
                    name="emb_weight", initializer=self.initializer
                ),
                is_sparse=False,
                dtype="uint16",
            )  # bfloat16
            cost = paddle.add(emb, label)
            avg_cost = paddle.mean(cost)

            sgd_optimizer = paddle.optimizer.SGD(
                learning_rate=self.learning_rate
            )
            sgd_optimizer = amp.bf16.decorate_bf16(
                sgd_optimizer,
                amp_lists=amp.bf16.AutoMixedPrecisionListsBF16(
                    custom_bf16_list={
                        'lookup_table',
                    }
                ),
                use_bf16_guard=False,
                use_pure_bf16=True,
            )
            sgd_optimizer.minimize(
                avg_cost, startup_program=fluid.default_startup_program()
            )

            train_reader = paddle.batch(self._data_reader, batch_size=1)
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            test_prog = main.clone(for_test=True)
            sgd_optimizer.amp_init(
                place, test_program=test_prog, use_bf16_test=True
            )

            ref_emb = np.full(self.w_shape, self.value, dtype=np.float32)
            ref_emb_bf16 = np.full(
                self.w_shape, self._fp322bf16(self.value), dtype=np.uint16
            )
            emb_weight = []

            for sample in train_reader():
                data = sample[0][0]
                label = sample[0][1]
                y_bf16 = convert_float_to_uint16(label)
                emb_weight = exe.run(
                    main,
                    feed={'X': data, 'Y': y_bf16},
                    fetch_list=['emb_weight'],
                )

                ref_emb = self._reference(data, ref_emb)
                ref_emb_bf16 = self._reference(data, ref_emb_bf16, True)

            self._check_output(emb_weight[0], ref_emb_bf16, bf16=True)
            self._check_output(emb_weight[0], ref_emb)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
