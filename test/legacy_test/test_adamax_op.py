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


def adamx_wrapper(
    param,
    grad,
    lr,
    moment,
    inf_norm,
    beta1_pow=None,
    master_weight=None,
    beta1=0.78,
    beta2=0.899,
    epsilon=1e-5,
    find_master=False,
):
    return paddle._C_ops.adamax_(
        param,
        grad,
        lr,
        moment,
        inf_norm,
        beta1_pow,
        master_weight,
        beta1,
        beta2,
        epsilon,
        find_master,
    )


class TestAdamaxOp1(OpTest):
    def setUp(self):
        '''Test Adamax Operator with supplied attributes'''
        self.op_type = "adamax"
        self.python_api = adamx_wrapper
        self.python_out_sig = ['Out']
        param = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        grad = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        moment = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        # The infinity norm is positive
        inf_norm = np.random.random((102, 105)).astype("float32")

        learning_rate = 0.002
        beta1 = 0.78
        beta2 = 0.899
        epsilon = 1e-5
        beta1_pow = beta1**10

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Moment': moment,
            'InfNorm': inf_norm,
            'LearningRate': np.array([learning_rate]).astype("float32"),
            'Beta1Pow': np.array([beta1_pow]).astype("float32"),
        }

        self.attrs = {'beta1': beta1, 'beta2': beta2, 'epsilon': epsilon}

        param_out, moment_out, inf_norm_out = adamax_step(
            self.inputs, self.attrs
        )

        self.outputs = {
            'ParamOut': param_out,
            'MomentOut': moment_out,
            'InfNormOut': inf_norm_out,
        }

    def test_check_output(self):
        self.check_output()


class TestAdamaxOp2(OpTest):
    '''Test Adamax Operator with default attributes'''

    def setUp(self):
        self.op_type = "adamax"
        self.python_api = adamx_wrapper
        self.python_out_sig = ['Out']
        param = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        grad = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        moment = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        # The infinity norm is positive
        inf_norm = np.random.random((102, 105)).astype("float32")

        learning_rate = 0.002
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        beta1_pow = beta1**8

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Moment': moment,
            'InfNorm': inf_norm,
            'LearningRate': np.array([learning_rate]).astype("float32"),
            'Beta1Pow': np.array([beta1_pow]).astype("float32"),
        }

        attrs = {'beta1': beta1, 'beta2': beta2, 'epsilon': epsilon}
        param_out, moment_out, inf_norm_out = adamax_step(self.inputs, attrs)

        self.outputs = {
            'ParamOut': param_out,
            'MomentOut': moment_out,
            'InfNormOut': inf_norm_out,
        }

    def test_check_output(self):
        self.check_output()


class TestAdamaxOpMultipleSteps(OpTest):
    def setUp(self):
        '''Test Adamax Operator with supplied attributes'''
        self.op_type = "adamax"
        self.python_api = adamx_wrapper
        self.python_out_sig = ['Out']
        self.num_steps = 10

        param = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        grad = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        moment = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        # The infinity norm is positive
        inf_norm = np.random.random((102, 105)).astype("float32")

        learning_rate = 0.002
        beta1 = 0.8
        beta2 = 0.99
        epsilon = 1e-5
        beta1_pow = 1

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Moment': moment,
            'InfNorm': inf_norm,
            'LearningRate': np.array([learning_rate]).astype("float32"),
            'Beta1Pow': np.array([beta1_pow]).astype("float32"),
        }

        self.attrs = {'beta1': beta1, 'beta2': beta2, 'epsilon': epsilon}

    def test_check_output(self):
        for _ in range(self.num_steps):
            param_out, moment_out, inf_norm_out = adamax_step(
                self.inputs, self.attrs
            )

            self.outputs = {
                'ParamOut': param_out,
                'MomentOut': moment_out,
                'InfNormOut': inf_norm_out,
            }

            # Verify output for this step
            self.check_output()

            # Output of this step becomes input for next step
            self.inputs['Param'] = param_out
            self.inputs['Moment'] = moment_out
            self.inputs['InfNorm'] = inf_norm_out

            # Update Beta1 Power accumulator for next step
            self.inputs['Beta1Pow'] *= self.attrs['beta1']

            # Randomize gradient for next step
            self.inputs['Grad'] = np.random.uniform(-1, 1, (102, 105)).astype(
                "float32"
            )


def adamax_step(inputs, attributes):
    '''
    Simulate one step of the adamax optimizer
    :param inputs: dict of inputs
    :param attributes: dict of attributes
    :return tuple: tuple of output param, moment, inf_norm and
    beta1 power accumulator
    '''
    param = inputs['Param']
    grad = inputs['Grad']
    moment = inputs['Moment']
    inf_norm = inputs['InfNorm']
    lr = inputs['LearningRate']
    beta1_pow = inputs['Beta1Pow']

    beta1 = attributes['beta1']
    beta2 = attributes['beta2']
    epsilon = attributes['epsilon']

    moment_out = beta1 * moment + (1 - beta1) * grad
    inf_norm_out = np.maximum(beta2 * inf_norm + epsilon, np.abs(grad))
    lr_t = lr / (1 - beta1_pow)
    param_out = param - lr_t * np.divide(moment_out, inf_norm_out)

    return param_out, moment_out, inf_norm_out


class TestAdamaxOpV2(unittest.TestCase):
    def test_adamax_op_invalid_input(self):
        import paddle

        paddle.disable_static()
        linear = paddle.nn.Linear(10, 10)
        with self.assertRaises(ValueError):
            adam = paddle.optimizer.Adamax(
                0.1, beta1=-1, parameters=linear.parameters()
            )
        with self.assertRaises(ValueError):
            adam = paddle.optimizer.Adamax(
                0.1, beta2=-1, parameters=linear.parameters()
            )
        with self.assertRaises(ValueError):
            adam = paddle.optimizer.Adamax(
                0.1, epsilon=-1, parameters=linear.parameters()
            )


class TestAdamaxOpMultiPrecision(unittest.TestCase):
    def _test_adamax_op_dygraph_place_amp(self, place, use_amp=False):
        import paddle

        paddle.disable_static()
        paddle.seed(10)
        paddle.set_device(place)
        input = paddle.randn((5, 5))

        model = paddle.nn.Linear(5, 5)
        optimizer = paddle.optimizer.Adamax(
            0.1, beta1=0.1, parameters=model.parameters()
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
                self._test_adamax_op_dygraph_place_amp(place, use_amp)


class TestAdamaxMultiPrecision2_0(unittest.TestCase):
    def dygraph_adamax_mp(self, mp, use_amp):
        paddle.disable_static()
        paddle.seed(100)
        paddle.set_device('gpu')
        input = paddle.randn((2, 2))
        model = paddle.nn.Linear(2, 2)
        optimizer = paddle.optimizer.Adamax(0.5, parameters=model.parameters())
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

    def static_adamax_mp(self, mp, use_amp):
        paddle.enable_static()
        paddle.seed(100)
        np.random.seed(100)
        exe = paddle.static.Executor('gpu')
        train_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        optimizer = paddle.optimizer.Adamax(0.1)
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
        output1_dy, params1_dy = self.dygraph_adamax_mp(use_amp=True, mp=True)
        output2_dy, params2_dy = self.dygraph_adamax_mp(use_amp=False, mp=False)
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
            output1_st = self.static_adamax_mp(use_amp=True, mp=True)
            output2_st = self.static_adamax_mp(use_amp=False, mp=False)
            for idx in range(len(output1_st)):
                np.testing.assert_allclose(
                    output1_st[idx].astype('float32'),
                    output2_st[idx].astype('float32'),
                    rtol=1e-05,
                    atol=0.1,
                )


if __name__ == "__main__":
    unittest.main()
