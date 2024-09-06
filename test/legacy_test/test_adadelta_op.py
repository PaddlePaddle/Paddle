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
from op_test import OpTest

import paddle
from paddle import base


def adadelta_wrapper(
    Param,
    Grad,
    AvgSquaredGrad,
    AvgSquaredUpdate,
    LearningRate,
    master_weight=None,
    rho=0.95,
    epsilon=1e-6,
):
    paddle._C_ops.adadelta_(
        Param,
        Grad,
        AvgSquaredGrad,
        AvgSquaredUpdate,
        LearningRate,
        None,
        rho,
        epsilon,
        False,
    )
    return Param, AvgSquaredGrad, AvgSquaredUpdate, LearningRate


class TestAdadeltaOp1(OpTest):
    def setUp(self):
        self.op_type = "adadelta"
        self.python_api = adadelta_wrapper
        self.python_out_sig = ['Out']
        param = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        grad = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        # The squared gradient is positive
        avg_squared_grad = np.random.random((102, 105)).astype("float32")
        # The squared update is positive
        avg_squared_update = np.random.random((102, 105)).astype("float32")

        rho = 0.95
        epsilon = 1e-6

        learning_rate = 1.0
        self.inputs = {
            'Param': param,
            'Grad': grad,
            'AvgSquaredGrad': avg_squared_grad,
            'AvgSquaredUpdate': avg_squared_update,
            'LearningRate': np.array([learning_rate]).astype("float32"),
        }

        self.attrs = {'rho': rho, 'epsilon': epsilon}

        avg_squared_grad_out = rho * avg_squared_grad + (1 - rho) * np.square(
            grad
        )
        update = -np.multiply(
            np.sqrt(
                np.divide(
                    avg_squared_update + epsilon, avg_squared_grad_out + epsilon
                )
            ),
            grad,
        )

        avg_squared_update_out = rho * avg_squared_update + (
            1 - rho
        ) * np.square(update)

        param_out = param + update

        self.outputs = {
            'ParamOut': param_out,
            'AvgSquaredGradOut': avg_squared_grad_out,
            'AvgSquaredUpdateOut': avg_squared_update_out,
        }

    def test_check_output(self):
        self.check_output()


class TestAdadeltaOp2(OpTest):
    '''Test Adadelta op with default attribute values'''

    def setUp(self):
        self.op_type = "adadelta"
        self.python_api = adadelta_wrapper
        self.python_out_sig = ['Out']
        param = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        grad = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        # The squared gradient is positive
        avg_squared_grad = np.random.random((102, 105)).astype("float32")
        # The squared update is positive
        avg_squared_update = np.random.random((102, 105)).astype("float32")

        rho = 0.95
        epsilon = 1e-6

        self.attrs = {'rho': rho, 'epsilon': epsilon}
        learning_rate = 1.0
        self.inputs = {
            'Param': param,
            'Grad': grad,
            'AvgSquaredGrad': avg_squared_grad,
            'AvgSquaredUpdate': avg_squared_update,
            'LearningRate': np.array([learning_rate]).astype("float32"),
        }

        avg_squared_grad_out = rho * avg_squared_grad + (1 - rho) * np.square(
            grad
        )
        update = -np.multiply(
            np.sqrt(
                np.divide(
                    avg_squared_update + epsilon, avg_squared_grad_out + epsilon
                )
            ),
            grad,
        )

        avg_squared_update_out = rho * avg_squared_update + (
            1 - rho
        ) * np.square(update)

        param_out = param + update

        self.outputs = {
            'ParamOut': param_out,
            'AvgSquaredGradOut': avg_squared_grad_out,
            'AvgSquaredUpdateOut': avg_squared_update_out,
        }

    def test_check_output(self):
        self.check_output()


class TestAdadeltaV2(unittest.TestCase):
    def test_adadelta_dygraph(self):
        paddle.disable_static(paddle.CPUPlace())
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear = paddle.nn.Linear(13, 5)
        # This can be any optimizer supported by dygraph.
        adam = paddle.optimizer.Adadelta(
            learning_rate=0.01,
            parameters=linear.parameters(),
            weight_decay=0.01,
        )
        out = linear(a)
        out.backward()
        adam.step()
        adam.clear_gradients()

    def test_adadelta(self):
        paddle.enable_static()
        place = base.CPUPlace()
        main = base.Program()
        with base.program_guard(main):
            x = paddle.static.data(name='x', shape=[-1, 13], dtype='float32')
            y = paddle.static.data(name='y', shape=[-1, 1], dtype='float32')
            y_predict = paddle.static.nn.fc(x, size=1)
            cost = paddle.nn.functional.square_error_cost(
                input=y_predict, label=y
            )
            avg_cost = paddle.mean(cost)

            rms_optimizer = paddle.optimizer.Adadelta(learning_rate=0.1)
            rms_optimizer.minimize(avg_cost)

            fetch_list = [avg_cost]
            train_reader = paddle.batch(
                paddle.dataset.uci_housing.train(), batch_size=1
            )
            feeder = base.DataFeeder(place=place, feed_list=[x, y])
            exe = base.Executor(place)
            exe.run(base.default_startup_program())
            for data in train_reader():
                exe.run(main, feed=feeder.feed(data), fetch_list=fetch_list)

    def test_raise_error(self):
        self.assertRaises(ValueError, paddle.optimizer.Adadelta, None)
        self.assertRaises(
            ValueError, paddle.optimizer.Adadelta, learning_rate=0.1, rho=None
        )
        self.assertRaises(
            ValueError,
            paddle.optimizer.Adadelta,
            learning_rate=0.1,
            epsilon=None,
        )


class TestAdadeltaV2Group(TestAdadeltaV2):
    def test_adadelta_dygraph(self):
        paddle.disable_static(paddle.CPUPlace())
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear_1 = paddle.nn.Linear(13, 5)
        linear_2 = paddle.nn.Linear(5, 5)
        # This can be any optimizer supported by dygraph.
        adam = paddle.optimizer.Adadelta(
            learning_rate=0.01,
            parameters=[
                {'params': linear_1.parameters()},
                {
                    'params': linear_2.parameters(),
                    'weight_decay': 0.001,
                },
            ],
            weight_decay=0.1,
        )
        out = linear_1(a)
        out = linear_2(out)
        out.backward()
        adam.step()
        adam.clear_gradients()


class TestAdadeltaOpMultiPrecision(unittest.TestCase):
    def _test_adadelta_op_dygraph_place_amp(self, place, use_amp=False):
        import paddle

        paddle.disable_static()
        paddle.seed(10)
        paddle.set_device(place)
        input = paddle.randn((5, 5))

        model = paddle.nn.Linear(5, 5)
        optimizer = paddle.optimizer.Adadelta(
            learning_rate=0.01,
            parameters=model.parameters(),
            weight_decay=0.1,
        )

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
                self._test_adadelta_op_dygraph_place_amp(place, use_amp)


class TestAdadeltaMultiPrecision2_0(unittest.TestCase):
    def dygraph_adadelta_mp(self, mp, use_amp):
        paddle.disable_static()
        paddle.seed(100)
        paddle.set_device('gpu')
        input = paddle.randn((2, 2))
        model = paddle.nn.Linear(2, 2)
        optimizer = paddle.optimizer.Adadelta(
            0.5, parameters=model.parameters()
        )
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

    def static_adadelta_mp(self, mp, use_amp):
        paddle.enable_static()
        paddle.seed(100)
        np.random.seed(100)
        exe = paddle.static.Executor('gpu')
        train_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        optimizer = paddle.optimizer.Adadelta(0.1)

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
        output1_dy, params1_dy = self.dygraph_adadelta_mp(use_amp=True, mp=True)
        output2_dy, params2_dy = self.dygraph_adadelta_mp(
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
            output1_st = self.static_adadelta_mp(use_amp=True, mp=True)
            output2_st = self.static_adadelta_mp(use_amp=False, mp=False)
            for idx in range(len(output1_st)):
                np.testing.assert_allclose(
                    output1_st[idx].astype('float32'),
                    output2_st[idx].astype('float32'),
                    rtol=1e-05,
                    atol=0.1,
                )


if __name__ == "__main__":
    unittest.main()
