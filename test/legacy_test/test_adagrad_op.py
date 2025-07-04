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

import math
import os
import unittest

import numpy as np
from op import Operator
from op_test import OpTest

import paddle
from paddle.base import core


def adamgrad_wrapper(
    param,
    grad,
    moment,
    learning_rate,
    master_weight=None,
    epsilon=1e-8,
    multi_precision=False,
):
    paddle._C_ops.adagrad_(
        param,
        grad,
        moment,
        learning_rate,
        master_weight,
        epsilon,
        multi_precision,
    )


class TestAdagradOp1(OpTest):
    '''Test Adagrad operator with explicit attributes'''

    def setUp(self):
        self.op_type = "adagrad"
        self.python_api = adamgrad_wrapper
        self.python_out_sig = ['out']
        param = np.random.random((123, 321)).astype("float32")
        grad = np.random.random((123, 321)).astype("float32")
        moment = np.zeros((123, 321)).astype("float32")
        lr = 0.01
        epsilon = 1e-8

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Moment': moment,
            'LearningRate': np.array([lr]).astype("float32"),
        }

        self.attrs = {'epsilon': epsilon}

        moment_out = moment + grad * grad
        param_out = param - lr * grad / (np.sqrt(moment_out) + epsilon)

        self.outputs = {'ParamOut': param_out, 'MomentOut': moment_out}

    def test_check_output(self):
        self.check_output()


class TestAdagradOp2(OpTest):
    '''Test Adagrad operator with default attributes'''

    def setUp(self):
        self.op_type = "adagrad"
        self.python_api = adamgrad_wrapper
        self.python_out_sig = ['out']

        param = np.random.random((123, 321)).astype("float32")
        grad = np.random.random((123, 321)).astype("float32")
        moment = np.zeros((123, 321)).astype("float32")
        lr = 0.01
        epsilon = 1e-6

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Moment': moment,
            'LearningRate': np.array([lr]).astype("float32"),
        }

        self.attrs = {'epsilon': epsilon, "multi_precision": False}

        moment_out = moment + grad * grad
        param_out = param - lr * grad / (np.sqrt(moment_out) + epsilon)

        self.outputs = {'ParamOut': param_out, 'MomentOut': moment_out}

    def test_check_output(self):
        self.check_output()


class TestSparseAdagradOp(unittest.TestCase):
    def check_with_place(self, place):
        scope = core.Scope()

        # create and initialize Grad Variable
        height = 10
        rows = [0, 4, 7, 4]
        row_numel = 12

        grad_selected_rows = scope.var('Grad').get_selected_rows()
        grad_selected_rows.set_height(height)
        grad_selected_rows.set_rows(rows)
        np_array = np.ones((len(rows), row_numel)).astype("float32")
        np_array[0, 0] = 2.0
        np_array[2, 8] = 4.0

        grad_tensor = grad_selected_rows.get_tensor()
        grad_tensor.set(np_array, place)

        # create and initialize Param Variable
        param = scope.var('Param').get_tensor()
        param_array = np.full((height, row_numel), 5.0).astype("float32")
        param.set(param_array, place)

        # create and initialize LearningRate Variable
        lr = scope.var('LearningRate').get_tensor()
        lr_array = np.full((1), 2.0).astype("float32")
        lr.set(lr_array, place)

        # create and initialize moment Variable
        moment = scope.var('Moment').get_tensor()
        moment_np_array = np.full((height, row_numel), 2.0).astype("float32")
        moment.set(moment_np_array, place)

        adagrad_op = Operator(
            "adagrad",
            Param='Param',
            Grad='Grad',
            ParamOut='Param',
            Moment='Moment',
            MomentOut='Moment',
            LearningRate='LearningRate',
            epsilon=2.0,
        )

        adagrad_op.run(scope, place)

        # get and compare moment result
        moment_result_array = np.array(moment)

        self.assertAlmostEqual(6.0, moment_result_array[rows[0], 0])
        self.assertAlmostEqual(3.0, moment_result_array[rows[0], 2])
        self.assertAlmostEqual(2.0, moment_result_array[1, 0])
        # 2.0 + (1.0 + 1.0)^2
        self.assertAlmostEqual(6.0, moment_result_array[rows[1], 10])
        self.assertAlmostEqual(6.0, moment_result_array[rows[3], 4])

        self.assertAlmostEqual(2.0, moment_result_array[5, 8])
        self.assertAlmostEqual(3.0, moment_result_array[rows[2], 1])
        self.assertAlmostEqual(18.0, moment_result_array[rows[2], 8])

        # get and compare param result
        result_array = np.array(param)

        def get_out(param, lr, grad, m, epsilon):
            return param - lr * grad / (math.sqrt(m) + epsilon)

        self.assertAlmostEqual(
            get_out(5.0, 2.0, 2.0, 6.0, 2.0), result_array[rows[0], 0], places=5
        )
        self.assertAlmostEqual(
            get_out(5.0, 2.0, 1.0, 3.0, 2.0), result_array[rows[0], 2], places=5
        )
        self.assertAlmostEqual(
            get_out(5.0, 2.0, 0.0, 2.0, 2.0), result_array[1, 0], places=5
        )

        # grad_merge = 1.0 + 1.0
        # m = 6.0
        self.assertAlmostEqual(
            get_out(5.0, 2.0, 2.0, 6.0, 2.0),
            result_array[rows[1], 10],
            places=5,
        )

        self.assertAlmostEqual(
            get_out(5.0, 2.0, 0.0, 2.0, 2.0), result_array[5, 8], places=5
        )
        self.assertAlmostEqual(
            get_out(5.0, 2.0, 1.0, 3.0, 2.0), result_array[rows[2], 1], places=5
        )
        self.assertAlmostEqual(
            get_out(5.0, 2.0, 4.0, 18.0, 2.0),
            result_array[rows[2], 8],
            places=5,
        )

    def test_sparse_adagrad(self):
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


class TestAdagradOpMultiPrecision(unittest.TestCase):
    def _test_adagrad_op_dygraph_place_amp(self, place, use_amp=False):
        import paddle

        paddle.disable_static()
        paddle.seed(10)
        paddle.set_device(place)
        input = paddle.randn((5, 5))

        model = paddle.nn.Linear(5, 5)
        optimizer = paddle.optimizer.Adagrad(0.1, parameters=model.parameters())
        optimizer._multi_precision = use_amp
        for idx in range(2):
            if place == 'gpu' and use_amp:
                model = paddle.amp.decorate(models=model, level='O2')
                scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

            if place == 'gpu' and use_amp:
                with paddle.amp.auto_cast(level='O2'):
                    output = model(input)
                    loss = paddle.mean(output)
                scaled = scaler.scale(loss)
                scaled.backward()
                scaler.step(optimizer)
                optimizer.clear_grad()
            else:
                output = model(input)
                loss = paddle.mean(output)
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()
        paddle.enable_static()

    def _get_places(self):
        import paddle

        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.is_compiled_with_cuda()
        ):
            places.append('cpu')
        if paddle.is_compiled_with_cuda():
            places.append('gpu')
        return places

    def test_main(self):
        for place in self._get_places():
            use_amp_list = [True, False]
            for use_amp in use_amp_list:
                self._test_adagrad_op_dygraph_place_amp(place, use_amp)


class TestAdagradMultiPrecision2_0(unittest.TestCase):
    def dygraph_adagrad_mp(self, mp, use_amp):
        paddle.disable_static()
        paddle.seed(100)
        paddle.set_device('gpu')
        input = paddle.randn((2, 2))
        model = paddle.nn.Linear(2, 2)
        optimizer = paddle.optimizer.Adagrad(0.5, parameters=model.parameters())
        optimizer._multi_precision = mp
        if use_amp:
            model = paddle.amp.decorate(models=model, level='O2')
            scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

        for idx in range(5):
            if use_amp:
                with paddle.amp.auto_cast(level='O2'):
                    output = model(input)
                    loss = paddle.mean(output)
                scaled = scaler.scale(loss)
                scaled.backward()
                scaler.minimize(optimizer, scaled)
                optimizer.clear_grad()
            else:
                output = model(input)
                loss = paddle.mean(output)
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()

        return output, model.parameters()

    def static_adagrad_mp(self, mp, use_amp):
        paddle.enable_static()
        paddle.seed(100)
        np.random.seed(100)
        exe = paddle.static.Executor('gpu')
        train_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        optimizer = paddle.optimizer.Adagrad(0.1)

        with paddle.static.program_guard(train_program, startup_program):
            if use_amp:
                data = paddle.static.data(
                    shape=[2, 2], name='X', dtype='float16'
                )
            else:
                data = paddle.static.data(
                    shape=[2, 2], name='X', dtype='float32'
                )
            hidden_layer = paddle.nn.Linear(2, 10)
            if use_amp:
                hidden_layer, optimizer = paddle.amp.decorate(
                    models=hidden_layer,
                    optimizers=optimizer,
                    level='O2',
                    master_weight=True,
                    master_grad=False,
                )
                with paddle.amp.auto_cast(
                    level='O2', dtype='float16', use_promote=True
                ):
                    hidden = hidden_layer(data)
                    loss = paddle.mean(hidden)
            else:
                hidden = hidden_layer(data)
                loss = paddle.mean(hidden)
            optimizer.minimize(loss)
        exe.run(startup_program)

        if use_amp:
            x = np.random.random(size=(2, 2)).astype('float16')
        else:
            x = np.random.random(size=(2, 2)).astype('float32')
        out = []
        for idx in range(5):
            (loss_data,) = exe.run(
                train_program, feed={"X": x}, fetch_list=[loss]
            )
            out.append(loss_data)
        return out

    def test_main(self):
        if not paddle.is_compiled_with_cuda():
            return
        "Test dygraph mode"
        output1_dy, params1_dy = self.dygraph_adagrad_mp(use_amp=True, mp=True)
        output2_dy, params2_dy = self.dygraph_adagrad_mp(
            use_amp=False, mp=False
        )
        np.testing.assert_allclose(
            output1_dy.astype('float32').numpy(),
            output2_dy.astype('float32').numpy(),
            rtol=1e-05,
            atol=0.1,
        )
        for idx in range(len(params1_dy)):
            np.testing.assert_allclose(
                params1_dy[idx].astype('float32').numpy(),
                params2_dy[idx].astype('float32').numpy(),
                rtol=1e-05,
                atol=0.1,
            )
        "Test static mode"
        with paddle.pir_utils.IrGuard():
            output1_st = self.static_adagrad_mp(use_amp=True, mp=True)
            output2_st = self.static_adagrad_mp(use_amp=False, mp=False)
            for idx in range(len(output1_st)):
                np.testing.assert_allclose(
                    output1_st[idx].astype('float32'),
                    output2_st[idx].astype('float32'),
                    rtol=1e-05,
                    atol=0.1,
                )


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
