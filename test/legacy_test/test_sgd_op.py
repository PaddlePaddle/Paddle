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

import os
import unittest

import numpy as np
from op import Operator
from op_test import OpTest
from utils import dygraph_guard

import paddle
from paddle.base import core

paddle.enable_static()


def sgd_wrapper(
    param, learning_rate, grad, master_param=None, multi_precision=False
):
    paddle._C_ops.sgd_(
        param, learning_rate, grad, master_param, multi_precision
    )


class TestSGDOp(OpTest):
    def setUp(self):
        self.op_type = "sgd"
        self.python_api = sgd_wrapper
        self.python_out_sig = ['Out']
        self.conf()
        w = np.random.random((self.h, self.w)).astype("float32")
        g = np.random.random((self.h, self.w)).astype("float32")
        lr = np.array([0.1]).astype("float32")

        self.inputs = {'Param': w, 'Grad': g, 'LearningRate': lr}
        self.outputs = {'ParamOut': w - lr * g}

    def conf(self):
        self.h = 102
        self.w = 105

    def test_check_output(self):
        self.check_output(check_pir=True)


class TestSGDOpCase8X(TestSGDOp):
    def conf(self):
        self.h = 10
        self.w = 64


class TestSparseSGDOp(unittest.TestCase):
    def check_with_place(self, place):
        scope = core.Scope()

        # create and initialize Grad Variable
        height = 10
        rows = [0, 4, 7]
        self.conf()

        grad_selected_rows = scope.var('Grad').get_selected_rows()
        grad_selected_rows.set_height(height)
        grad_selected_rows.set_rows(rows)
        np_array = np.ones((len(rows), self.row_numel)).astype("float32")
        np_array[0, 0] = 2.0
        np_array[2, 8] = 4.0

        grad_tensor = grad_selected_rows.get_tensor()
        grad_tensor.set(np_array, place)

        # create and initialize Param Variable
        param = scope.var('Param').get_tensor()
        param_array = np.full((height, self.row_numel), 5.0).astype("float32")
        param.set(param_array, place)

        # create and initialize LearningRate Variable
        lr = scope.var('LearningRate').get_tensor()
        lr_array = np.full((1), 2.0).astype("float32")
        lr.set(lr_array, place)

        # create and run sgd operator
        sgd_op = Operator(
            "sgd",
            Param='Param',
            Grad='Grad',
            ParamOut='Param',
            LearningRate='LearningRate',
        )
        sgd_op.run(scope, place)

        # get and compare result
        result_array = np.array(param)

        # rows[0] = 0, 5.0 - 2.0 * 2.0
        self.assertAlmostEqual(1.0, result_array[rows[0], 0])
        # rows[0] = 0, 5.0 - 2.0 * 1.0
        self.assertAlmostEqual(3.0, result_array[rows[0], 2])
        # 5.0 - 2.0 * 0.0
        self.assertAlmostEqual(5.0, result_array[1, 0])
        # rows[1] = 4, 5.0 - 2.0 * 1.0
        self.assertAlmostEqual(3.0, result_array[rows[1], 10])
        # 5.0 - 2.0 * 0.0
        self.assertAlmostEqual(5.0, result_array[5, 8])
        # rows[2] = 7, 5.0 - 2.0 * 1.0
        self.assertAlmostEqual(3.0, result_array[rows[2], 1])
        # rows[2] = 7, 5.0 - 2.0 * 4.0
        self.assertAlmostEqual(-3.0, result_array[rows[2], 8])

    def test_sparse_sgd(self):
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(core.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        for place in places:
            self.check_with_place(place)

    def conf(self):
        self.row_numel = 12


class TestSparseSGDOpCase8X(TestSparseSGDOp):
    def conf(self):
        self.row_numel = 16


class TestSGDOpOptimizeSelectedRows(unittest.TestCase):
    def check_with_place(self, place):
        scope = core.Scope()

        row_width = 12
        # create and initialize Grad Variable
        grad_height = 10
        grad_rows = [0, 4, 7]

        grad_selected_rows = scope.var('Grad').get_selected_rows()
        grad_selected_rows.set_height(grad_height)
        grad_selected_rows.set_rows(grad_rows)
        grad_array = np.ones((len(grad_rows), row_width)).astype("float32")
        grad_array[0, 0] = 2.0
        grad_array[2, 8] = 4.0

        grad_tensor = grad_selected_rows.get_tensor()
        grad_tensor.set(grad_array, place)

        # create and initialize Param Variable
        # create and initialize W Variable
        param_rows = [0, 1, 2, 3, 4, 5, 6, 7]

        # init Param
        w_selected_rows = scope.var('Param').get_selected_rows()
        w_selected_rows.set_height(len(param_rows))
        w_selected_rows.set_rows(param_rows)
        w_selected_rows.sync_index()
        w_array = np.ones((len(param_rows), row_width)).astype("float32")
        for i in range(len(param_rows)):
            w_array[i] *= i
        w_tensor = w_selected_rows.get_tensor()
        w_tensor.set(w_array, place)

        w_before_optimize = np.array(w_tensor)

        # create and initialize LearningRate Variable
        lr_value = 0.1
        lr = scope.var('LearningRate').get_tensor()
        lr_array = np.full((1), lr_value).astype("float32")
        lr.set(lr_array, place)

        # optimize with Python
        w_after_optimize = np.copy(w_before_optimize)
        for index, id in enumerate(grad_rows):
            w_after_optimize[id] = (
                w_before_optimize[id] - lr_value * grad_array[index]
            )

        # create and run sgd operator
        sgd_op = Operator(
            "sgd",
            Param='Param',
            Grad='Grad',
            ParamOut='Param',
            LearningRate='LearningRate',
        )
        sgd_op.run(scope, place)

        # get and compare result
        result_array = np.array(w_tensor)
        assert (result_array == w_after_optimize).all()

    def test_sparse_parameter_sgd(self):
        places = [core.CPUPlace()]
        # do not support GPU kernel currently
        for place in places:
            self.check_with_place(place)


class TestSGDV2(unittest.TestCase):
    def test_sgd_dygraph(self):
        paddle.disable_static()
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear = paddle.nn.Linear(13, 5)
        # This can be any optimizer supported by dygraph.
        adam = paddle.optimizer.SGD(
            learning_rate=0.01,
            parameters=linear.parameters(),
            weight_decay=0.01,
        )
        out = linear(a)
        out.backward()
        adam.step()
        adam.clear_gradients()

    def test_raise_error(self):
        self.assertRaises(ValueError, paddle.optimizer.SGD, learning_rate=None)

    def test_sgd_group_dygraph(self):
        paddle.disable_static()
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear_1 = paddle.nn.Linear(13, 5)
        linear_2 = paddle.nn.Linear(5, 3)
        # This can be any optimizer supported by dygraph.
        adam = paddle.optimizer.SGD(
            learning_rate=0.01,
            parameters=[
                {'params': linear_1.parameters()},
                {
                    'params': linear_2.parameters(),
                    'weight_decay': 0.001,
                    'learning_rate': 0.1,
                },
            ],
            weight_decay=0.01,
        )
        out = linear_1(a)
        out = linear_2(out)
        out.backward()
        adam.step()
        adam.clear_gradients()

    def test_weight_decay_int(self):
        paddle.disable_static()
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear = paddle.nn.Linear(13, 5)
        # This can be any optimizer supported by dygraph.
        adam = paddle.optimizer.SGD(
            learning_rate=0.01,
            parameters=linear.parameters(),
            weight_decay=1,
        )
        out = linear(a)
        out.backward()
        adam.step()
        adam.clear_gradients()


class TestSGDSimple(unittest.TestCase):
    def setUp(self) -> None:
        self.data = np.random.random(size=(2, 2)).astype('float32')

    def run_static(self):
        with paddle.pir_utils.IrGuard():
            paddle.seed(10)
            np.random.seed(10)

            exe = paddle.static.Executor('gpu')
            train_program = paddle.static.Program()
            startup_program = paddle.static.Program()

            with paddle.static.program_guard(train_program, startup_program):
                input = paddle.static.data(
                    shape=[2, 2], name='input', dtype='float32'
                )
                model = paddle.nn.Linear(2, 2)
                output = model(input)
                loss = paddle.mean(output)

                optimizer = paddle.optimizer.SGD()
                optimizer.minimize(loss)

            exe.run(startup_program)

            out = []
            for _ in range(5):
                (loss_data,) = exe.run(
                    train_program, feed={"input": self.data}, fetch_list=[loss]
                )
                out.append(loss_data)
            return out

    def run_dygraph(self):
        with dygraph_guard():
            paddle.seed(10)
            np.random.seed(10)

            out = []
            model = paddle.nn.Linear(2, 2)
            optimizer = paddle.optimizer.SGD(parameters=model.parameters())
            for _ in range(5):
                output = model(paddle.to_tensor(self.data))
                loss = paddle.mean(output)
                out.append(loss.numpy())
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()

            return out

    def test_main(self):
        if not paddle.is_compiled_with_cuda():
            return
        out1 = self.run_dygraph()
        out2 = self.run_static()
        np.testing.assert_allclose(out1, out2)


if __name__ == "__main__":
    unittest.main()
