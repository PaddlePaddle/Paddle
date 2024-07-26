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

import paddle
from paddle import base
from paddle.base import core


def adam_wrapper(
    param,
    grad,
    LearningRate,
    moment1,
    moment2,
    beta1_pow,
    beta2_pow,
    master_weight=None,
    find_inf=None,
    beta1=0.78,
    beta2=0.836,
    epsilon=1e-4,
    lazy_mode=False,
):
    _, _, _, _, _, _ = paddle._C_ops.adam_(
        param,
        grad,
        LearningRate,
        moment1,
        moment2,
        beta1_pow,
        beta2_pow,
        master_weight,
        find_inf,
        beta1,
        beta2,
        epsilon,
        lazy_mode,
        1000,
        False,
        False,
    )


class TestAdamOp1(OpTest):
    def setUp(self):
        '''Test Adam Op with supplied attributes'''
        self.op_type = "adam"
        self.python_api = adam_wrapper
        self.python_out_sig = ['Out']
        param = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        grad = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        moment1 = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        # The second moment is positive
        moment2 = np.random.random((102, 105)).astype("float32")

        learning_rate = 0.004
        beta1 = 0.78
        beta2 = 0.836
        epsilon = 1e-4
        beta1_pow = beta1**10
        beta2_pow = beta2**10

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Moment1': moment1,
            'Moment2': moment2,
            'LearningRate': np.array([learning_rate]).astype("float32"),
            'Beta1Pow': np.array([beta1_pow]).astype("float32"),
            'Beta2Pow': np.array([beta2_pow]).astype("float32"),
        }

        self.attrs = {'epsilon': epsilon, 'beta1': beta1, 'beta2': beta2}

        param_out, moment1_out, moment2_out = adam_step(self.inputs, self.attrs)

        self.outputs = {
            'Moment1Out': moment1_out,
            'Moment2Out': moment2_out,
            'ParamOut': param_out,
            'Beta1PowOut': np.array([beta1_pow]).astype("float32") * beta1,
            'Beta2PowOut': np.array([beta2_pow]).astype("float32") * beta2,
        }

    def test_check_output(self):
        self.check_output(check_pir=True)


class TestAdamOp2(OpTest):
    def set_shape(self):
        self.shape = (102, 105)

    def setUp(self):
        '''Test Adam Op with supplied attributes'''
        self.op_type = "adam"
        self.python_api = adam_wrapper
        self.python_out_sig = ['Out']
        self.set_shape()
        param = np.random.uniform(-1, 1, self.shape).astype("float32")
        grad = np.random.uniform(-1, 1, self.shape).astype("float32")
        moment1 = np.random.uniform(-1, 1, self.shape).astype("float32")
        # The second moment is positive
        moment2 = np.random.random(self.shape).astype("float32")

        learning_rate = 0.001
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        beta1_pow = beta1**10
        beta2_pow = beta2**10

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Moment1': moment1,
            'Moment2': moment2,
            'LearningRate': np.array([learning_rate]).astype("float32"),
            'Beta1Pow': np.array([beta1_pow]).astype("float32"),
            'Beta2Pow': np.array([beta2_pow]).astype("float32"),
        }

        attributes = {'epsilon': epsilon, 'beta1': beta1, 'beta2': beta2}

        param_out, moment1_out, moment2_out = adam_step(self.inputs, attributes)

        self.outputs = {
            'Moment1Out': moment1_out,
            'Moment2Out': moment2_out,
            'ParamOut': param_out,
            'Beta1PowOut': np.array([beta1_pow]).astype("float32") * beta1,
            'Beta2PowOut': np.array([beta2_pow]).astype("float32") * beta2,
        }

    def test_check_output(self):
        self.check_output(check_pir=True)


class TestAdamOnlyTailOp(TestAdamOp2):
    def set_shape(self):
        self.shape = 3


class TestAdamOpMultipleSteps(OpTest):
    def setUp(self):
        '''Test Adam Operator with supplied attributes'''
        self.op_type = "adam"
        self.python_api = adam_wrapper
        self.python_out_sig = ['Out']
        self.num_steps = 10

        param = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        grad = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        moment1 = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        # The second moment is positive
        moment2 = np.random.random((102, 105)).astype("float32")

        learning_rate = 0.001
        self.beta1 = 0.9
        self.beta2 = 0.999
        epsilon = 1e-8
        self.beta1_pow = self.beta1**10
        self.beta2_pow = self.beta2**10

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Moment1': moment1,
            'Moment2': moment2,
            'LearningRate': np.array([learning_rate]).astype("float32"),
            'Beta1Pow': np.array([self.beta1_pow]).astype("float32"),
            'Beta2Pow': np.array([self.beta2_pow]).astype("float32"),
        }

        self.attrs = {
            'epsilon': epsilon,
            'beta1': self.beta1,
            'beta2': self.beta2,
        }

    def test_check_output(self):
        for _ in range(self.num_steps):
            param_out, moment1_out, moment2_out = adam_step(
                self.inputs, self.attrs
            )

            beta1_pow_out = self.inputs['Beta1Pow'] * self.beta1
            beta2_pow_out = self.inputs['Beta2Pow'] * self.beta2
            self.outputs = {
                'Moment1Out': moment1_out,
                'Moment2Out': moment2_out,
                'ParamOut': param_out,
                'Beta1PowOut': beta1_pow_out,
                'Beta2PowOut': beta2_pow_out,
            }

            # Verify output for this step
            self.check_output(check_pir=True)

            # Output of this step becomes input for next step
            self.inputs['Param'] = param_out
            self.inputs['Moment1'] = moment1_out
            self.inputs['Moment2'] = moment2_out

            # Update powers of Beta1 and Beta2 for next time step
            self.inputs['Beta1Pow'] = beta1_pow_out
            self.inputs['Beta2Pow'] = beta2_pow_out

            # Randomize gradient for next step
            self.inputs['Grad'] = np.random.uniform(-1, 1, (102, 105)).astype(
                "float32"
            )


def adam_step(inputs, attributes):
    '''
    Simulate one step of the adam optimizer
    :param inputs: dict of inputs
    :param attributes: dict of attributes
    :return tuple: tuple of output param, moment1, moment2,
    beta1 power accumulator and beta2 power accumulator
    '''
    param = inputs['Param']
    grad = inputs['Grad']
    moment1 = inputs['Moment1']
    moment2 = inputs['Moment2']
    lr = inputs['LearningRate']
    beta1_pow = inputs['Beta1Pow']
    beta2_pow = inputs['Beta2Pow']

    epsilon = attributes['epsilon']

    if 'beta1' in attributes:
        beta1 = attributes['beta1']
    else:
        beta1 = inputs['Beta1Tensor'][0]
    if 'beta2' in attributes:
        beta2 = attributes['beta2']
    else:
        beta2 = inputs['Beta2Tensor'][0]

    moment1_out = beta1 * moment1 + (1 - beta1) * grad
    moment2_out = beta2 * moment2 + (1 - beta2) * np.square(grad)
    lr_t = lr * np.sqrt(1 - beta2_pow) / (1 - beta1_pow)
    param_out = param - lr_t * (moment1_out / (np.sqrt(moment2_out) + epsilon))
    return param_out, moment1_out, moment2_out


def adamw_step(inputs, attributes):
    '''
    Simulate one step of the adam optimizer
    :param inputs: dict of inputs
    :param attributes: dict of attributes
    :return tuple: tuple of output param, moment1, moment2,
    beta1 power accumulator and beta2 power accumulator
    '''
    param = inputs['Param']
    grad = inputs['Grad']
    moment1 = inputs['Moment1']
    moment2 = inputs['Moment2']
    lr = inputs['LearningRate']
    beta1_pow = inputs['Beta1Pow']
    beta2_pow = inputs['Beta2Pow']

    epsilon = attributes['epsilon']
    coeff = attributes["coeff"]
    if attributes.get("with_decay", False):
        decay = 1.0 - lr * coeff
        param2 = param * decay
        param = param2.copy()
    if 'beta1' in attributes:
        beta1 = attributes['beta1']
    else:
        beta1 = inputs['Beta1Tensor'][0]
    if 'beta2' in attributes:
        beta2 = attributes['beta2']
    else:
        beta2 = inputs['Beta2Tensor'][0]

    moment1_out = beta1 * moment1 + (1 - beta1) * grad
    moment2_out = beta2 * moment2 + (1 - beta2) * np.square(grad)
    lr_t = lr * np.sqrt(1 - beta2_pow) / (1 - beta1_pow)
    param_out = param - lr_t * (moment1_out / (np.sqrt(moment2_out) + epsilon))

    return param_out, moment1_out, moment2_out


def adam_step_sparse(
    inputs, attributes, height, rows, row_numel, np_grad, lazy_mode
):
    '''
    Simulate one step of the adam optimizer
    :param inputs: dict of inputs
    :param attributes: dict of attributes
    :return tuple: tuple of output param, moment1, moment2,
    beta1 power accumulator and beta2 power accumulator
    '''
    param = inputs['Param']
    # grad = inputs['Grad']
    moment1 = inputs['Moment1']
    moment2 = inputs['Moment2']
    lr = inputs['LearningRate']
    beta1_pow = inputs['Beta1Pow']
    beta2_pow = inputs['Beta2Pow']

    beta1 = attributes['beta1']
    beta2 = attributes['beta2']
    epsilon = attributes['epsilon']

    moment1_out = np.zeros(shape=[height, row_numel])
    moment2_out = np.zeros(shape=[height, row_numel])
    param_out = np.zeros(shape=[height, row_numel])

    def update_row(row_id, update_value):
        moment1_out[row_id] = (
            beta1 * moment1[row_id] + (1 - beta1) * update_value
        )
        moment2_out[row_id] = beta2 * moment2[row_id] + (1 - beta2) * np.square(
            update_value
        )
        lr_t = lr * np.sqrt(1 - beta2_pow) / (1 - beta1_pow)
        param_out[row_id] = param[row_id] - lr_t * (
            moment1_out[row_id] / (np.sqrt(moment2_out[row_id]) + epsilon)
        )

    if lazy_mode:
        for idx, row_id in enumerate(rows):
            update_row(row_id, np_grad[idx])
    else:
        for row_id in range(param_out.shape[0]):
            update_value = np.zeros(np_grad[0].shape).astype("float32")
            if row_id in rows:
                update_value = np_grad[rows.index(row_id)]
            update_row(row_id, update_value)

    return param_out, moment1_out, moment2_out


class TestSparseAdamOp(unittest.TestCase):
    def setup(self, scope, place, lazy_mode):
        beta1 = 0.78
        beta2 = 0.836
        epsilon = 1e-4
        beta1_pow = np.array([beta1**10]).astype("float32")
        beta2_pow = np.array([beta2**10]).astype("float32")

        height = 10
        rows = [0, 4, 7]
        self.rows = rows
        row_numel = 12
        self.row_numel = row_numel
        self.dense_inputs = {
            "Param": np.full((height, row_numel), 5.0).astype("float32"),
            "Moment1": np.full((height, row_numel), 5.0).astype("float32"),
            "Moment2": np.full((height, row_numel), 5.0).astype("float32"),
            'Beta1Pow': beta1_pow,
            'Beta2Pow': beta2_pow,
            "LearningRate": np.full((1), 2.0).astype("float32"),
        }
        self.init_output = np.full((height, row_numel), 0.0).astype("float32")
        self.attrs = {
            'epsilon': epsilon,
            'beta1': beta1,
            'beta2': beta2,
            'min_row_size_to_use_multithread': 2,
        }

        grad_selected_rows = scope.var('Grad').get_selected_rows()
        grad_selected_rows.set_height(height)
        grad_selected_rows.set_rows(rows)
        np_array = np.ones((len(rows), row_numel)).astype("float32")
        np_array[0, 0] = 2.0
        np_array[2, 8] = 4.0

        grad_tensor = grad_selected_rows.get_tensor()
        grad_tensor.set(np_array, place)

        self.sparse_inputs = ["Grad"]

        param_out, mom1, mom2 = adam_step_sparse(
            self.dense_inputs,
            self.attrs,
            height,
            rows,
            row_numel,
            np_array,
            lazy_mode,
        )
        self.outputs = {
            "ParamOut": param_out,
            "Moment1Out": mom1,
            "Moment2Out": mom2,
            'Beta1PowOut': beta1_pow * beta1,
            'Beta2PowOut': beta2_pow * beta2,
        }

    def check_with_place(self, place, lazy_mode):
        scope = core.Scope()
        self.setup(scope, place, lazy_mode)

        op_args = {}
        op_args['lazy_mode'] = lazy_mode
        for key, np_array in self.dense_inputs.items():
            var = scope.var(key).get_tensor()
            var.set(np_array, place)
            op_args[key] = key
        for s in self.sparse_inputs:
            op_args[s] = s
        for s in self.outputs:
            var = scope.var(s).get_tensor()
            var.set(self.init_output, place)
            op_args[s] = s
        for k in self.attrs:
            op_args[k] = self.attrs[k]

        # create and run sgd operator
        adam_op = Operator("adam", **op_args)
        adam_op.run(scope, place)

        for key, np_array in self.outputs.items():
            out_var = scope.var(key).get_tensor()
            actual = np.array(out_var)
            actual = actual.reshape([actual.size])
            np_array = np_array.reshape([np_array.size])

            for i in range(np_array.size):
                self.assertLess((actual[i] - np_array[i]), 0.00001)

    def test_sparse_adam(self):
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
            for lazy_mode in (True, False):
                self.check_with_place(place, lazy_mode)


class TestAdamOpBetaVariable(OpTest):
    def setUp(self):
        '''Test Adam Op with beta as Variable'''
        self.op_type = "adam"
        self.python_api = adam_wrapper
        self.python_out_sig = ['Out']
        param = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        grad = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        moment1 = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        # The second moment is positive
        moment2 = np.random.random((102, 105)).astype("float32")
        beta1 = 0.85
        beta2 = 0.95

        learning_rate = 0.001
        epsilon = 1e-8
        beta1_pow = beta1**10
        beta2_pow = beta2**10

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Moment1': moment1,
            'Moment2': moment2,
            'LearningRate': np.array([learning_rate]).astype("float32"),
            'Beta1Pow': np.array([beta1_pow]).astype("float32"),
            'Beta2Pow': np.array([beta2_pow]).astype("float32"),
            "Beta1Tensor": np.array([beta1]).astype("float32"),
            "Beta2Tensor": np.array([beta2]).astype("float32"),
        }

        attributes = {'epsilon': epsilon}

        param_out, moment1_out, moment2_out = adam_step(self.inputs, attributes)

        self.outputs = {
            'Moment1Out': moment1_out,
            'Moment2Out': moment2_out,
            'ParamOut': param_out,
            'Beta1PowOut': np.array([beta1_pow]).astype("float32") * beta1,
            'Beta2PowOut': np.array([beta2_pow]).astype("float32") * beta2,
        }

    def test_check_output(self):
        self.check_output(check_pir=True)


class TestAdamOpBetaEpsilonVariable(OpTest):
    def setUp(self):
        '''Test Adam Op with beta/epsilon as Variable'''
        self.op_type = "adam"
        self.python_api = adam_wrapper
        self.python_out_sig = ['Out']
        param = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        grad = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        moment1 = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        # The second moment is positive
        moment2 = np.random.random((102, 105)).astype("float32")
        beta1 = 0.85
        beta2 = 0.95

        learning_rate = 0.001
        epsilon = 1e-8
        beta1_pow = beta1**10
        beta2_pow = beta2**10

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Moment1': moment1,
            'Moment2': moment2,
            'LearningRate': np.array([learning_rate]).astype("float32"),
            'Beta1Pow': np.array([beta1_pow]).astype("float32"),
            'Beta2Pow': np.array([beta2_pow]).astype("float32"),
            "Beta1Tensor": np.array([beta1]).astype("float32"),
            "Beta2Tensor": np.array([beta2]).astype("float32"),
            "EpsilonTensor": np.array([epsilon]).astype("float32"),
        }

        attributes = {'epsilon': epsilon}

        param_out, moment1_out, moment2_out = adam_step(self.inputs, attributes)

        self.outputs = {
            'Moment1Out': moment1_out,
            'Moment2Out': moment2_out,
            'ParamOut': param_out,
            'Beta1PowOut': np.array([beta1_pow]).astype("float32") * beta1,
            'Beta2PowOut': np.array([beta2_pow]).astype("float32") * beta2,
        }

    def test_check_output(self):
        self.check_output(check_pir=True)


class TestAdamOpWithGlobalBetaPow(OpTest):
    def setUp(self):
        '''Test Adam Op with global_beta_pow'''
        self.op_type = "adam"
        self.python_api = adam_wrapper
        self.python_out_sig = ['Out']
        param = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        grad = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        moment1 = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        # The second moment is positive
        moment2 = np.random.random((102, 105)).astype("float32")
        beta1 = 0.85
        beta2 = 0.95

        learning_rate = 0.001
        epsilon = 1e-8
        beta1_pow = beta1**10
        beta2_pow = beta2**10

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Moment1': moment1,
            'Moment2': moment2,
            'LearningRate': np.array([learning_rate]).astype("float32"),
            'Beta1Pow': np.array([beta1_pow]).astype("float32"),
            'Beta2Pow': np.array([beta2_pow]).astype("float32"),
            "Beta1Tensor": np.array([beta1]).astype("float32"),
            "Beta2Tensor": np.array([beta2]).astype("float32"),
            "EpsilonTensor": np.array([epsilon]).astype("float32"),
        }

        attributes = {'epsilon': epsilon}

        param_out, moment1_out, moment2_out = adam_step(self.inputs, attributes)

        self.attrs = {'use_global_beta_pow': True}

        # use_global_beta_pow=True, Beta1PowOut and Beta2PowOut are empty.
        self.outputs = {
            'Moment1Out': moment1_out,
            'Moment2Out': moment2_out,
            'ParamOut': param_out,
            'Beta1PowOut': np.array([]),
            'Beta2PowOut': np.array([]),
        }

    def test_check_output(self):
        self.check_output(check_pir=True)


class TestAdamOpWithSkipUpdate(OpTest):
    def setUp(self):
        '''Test Adam Op with global_beta_pow'''
        self.op_type = "adam"
        self.python_api = adam_wrapper
        self.python_out_sig = ['Out']
        param = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        grad = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        moment1 = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        # The second moment is positive
        moment2 = np.random.random((102, 105)).astype("float32")
        beta1 = 0.85
        beta2 = 0.95

        learning_rate = 0.001
        epsilon = 1e-8
        beta1_pow = beta1**10
        beta2_pow = beta2**10

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Moment1': moment1,
            'Moment2': moment2,
            'LearningRate': np.array([learning_rate]).astype("float32"),
            'Beta1Pow': np.array([beta1_pow]).astype("float32"),
            'Beta2Pow': np.array([beta2_pow]).astype("float32"),
            "Beta1Tensor": np.array([beta1]).astype("float32"),
            "Beta2Tensor": np.array([beta2]).astype("float32"),
            "EpsilonTensor": np.array([epsilon]).astype("float32"),
            "SkipUpdate": np.array([True]).astype("bool"),
        }

        attributes = {'epsilon': epsilon}

        self.attrs = {'use_global_beta_pow': True}

        # use_global_beta_pow=True, Beta1PowOut and Beta2PowOut are empty.
        self.outputs = {
            'Moment1Out': moment1,
            'Moment2Out': moment2,
            'ParamOut': param,
            'Beta1PowOut': np.array([]),
            'Beta2PowOut': np.array([]),
        }

    def test_check_output(self):
        self.check_output(check_pir=True)


class TestAdamOpV2(unittest.TestCase):
    def test_pir_adam_op(self):
        with paddle.pir_utils.IrGuard():
            place = base.CPUPlace()
            shape = [2, 3, 8, 8]
            exe = base.Executor(place)
            train_prog = paddle.static.Program()
            startup = paddle.static.Program()
            with paddle.static.program_guard(train_prog, startup):
                with base.unique_name.guard():
                    data = paddle.static.data(name="data", shape=shape)
                    conv_layer = paddle.nn.Conv2D(3, 8, 3)
                    conv = conv_layer(data)
                    loss = paddle.mean(conv)

                    beta1 = paddle.pir.core.create_parameter(
                        'float32',
                        [1],
                        initializer=paddle.nn.initializer.Constant(0.85),
                    )
                    beta2 = paddle.pir.core.create_parameter(
                        'float32',
                        [1],
                        initializer=paddle.nn.initializer.Constant(0.95),
                    )
                    betas = [beta1, beta2]
                    opt = paddle.optimizer.Adam(
                        learning_rate=1e-5,
                        beta1=beta1,
                        beta2=beta2,
                        weight_decay=0.01,
                        epsilon=1e-8,
                    )
                    opt.minimize(loss)

            exe.run(startup)
            data_np = np.random.random(shape).astype('float32')
            rets = exe.run(
                train_prog, feed={"data": data_np}, fetch_list=[loss]
            )
            assert rets[0] is not None

    def test_adam_op_dygraph(self):
        paddle.disable_static()
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear = paddle.nn.Linear(13, 5)

        adam = paddle.optimizer.Adam(
            learning_rate=0.01, parameters=linear.parameters()
        )
        out = linear(a)
        out.backward()
        adam.step()
        adam.clear_gradients()
        paddle.enable_static()

    def test_adam_op_with_state_dict(self):
        paddle.disable_static()
        emb = paddle.nn.Embedding(10, 10)

        adam = paddle.optimizer.Adam(0.001, parameters=emb.parameters())
        state_dict = adam.state_dict()
        adam.set_state_dict(state_dict)

        # learning_rate is LRScheduler
        learning_rate = paddle.optimizer.lr.CosineAnnealingDecay(
            learning_rate=0.1, T_max=10
        )
        adam = paddle.optimizer.Adam(
            learning_rate=learning_rate,
            weight_decay=paddle.regularizer.L2Decay(0.001),
            parameters=emb.parameters(),
        )
        lr = adam.get_lr()
        state_dict = adam.state_dict()
        adam.set_state_dict(state_dict)

        # learning_rate is Tensor
        with self.assertRaises(TypeError):
            learning_rate = np.array([0.01]).astype("float32")
            learning_rate = paddle.to_tensor(learning_rate)
            adam = paddle.optimizer.Adam(
                learning_rate=learning_rate, parameters=emb.parameters()
            )

        params = adam.get_opti_var_name_list()
        assert params is not None
        paddle.enable_static()

    def test_adam_with_grad_clip(self):
        paddle.disable_static()
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear = paddle.nn.Linear(13, 5)
        clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
        adam = paddle.optimizer.Adam(
            0.1, parameters=linear.parameters(), grad_clip=clip
        )
        out = linear(a)
        out.backward()
        adam.step()
        adam.clear_gradients()
        paddle.enable_static()

    def test_adam_op_with_set_lr(self):
        paddle.disable_static()
        linear = paddle.nn.Linear(10, 10)
        adam = paddle.optimizer.Adam(0.1, parameters=linear.parameters())

        lr = 0.01
        adam.set_lr(lr)
        cur_lr = adam.get_lr()
        assert lr == cur_lr
        with self.assertRaises(TypeError):
            lr_var = paddle.static.create_global_var(
                shape=[1], value=lr, dtype='float32'
            )
            adam.set_lr(lr_var)
        paddle.enable_static()

    def test_adam_op_invalid_input(self):
        paddle.disable_static()
        linear = paddle.nn.Linear(10, 10)
        with self.assertRaises(ValueError):
            adam = paddle.optimizer.Adam(
                0.1, beta1=-1, parameters=linear.parameters()
            )
        with self.assertRaises(ValueError):
            adam = paddle.optimizer.Adam(
                0.1, beta2=-1, parameters=linear.parameters()
            )
        with self.assertRaises(ValueError):
            adam = paddle.optimizer.Adam(
                0.1, epsilon=-1, parameters=linear.parameters()
            )
        paddle.enable_static()

    def test_adam_op_with_sparse_input_and_weight_decay(self):
        paddle.disable_static()
        x_data = np.arange(0, 10).reshape((10, 1)).astype(np.int64)
        x = paddle.to_tensor(x_data, stop_gradient=False)
        emb = paddle.nn.Embedding(10, 10, sparse=True)
        adam = paddle.optimizer.Adam(
            0.001, parameters=emb.parameters(), weight_decay=0.01
        )

        with self.assertRaises(RuntimeError):
            out = emb(x)
            out.backward()
            adam.step()
        paddle.enable_static()


class TestAdamOpV2Group(TestAdamOpV2):
    def test_adam_op(self):
        paddle.disable_static()
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear_1 = paddle.nn.Linear(13, 5)
        linear_2 = paddle.nn.Linear(5, 3)
        # This can be any optimizer supported by dygraph.
        adam = paddle.optimizer.Adam(
            learning_rate=0.01,
            parameters=[
                {'params': linear_1.parameters()},
                {
                    'params': linear_2.parameters(),
                    'weight_decay': 0.001,
                    'beta1': 0.1,
                    'beta2': 0.99,
                },
            ],
            weight_decay=0.1,
        )
        out = linear_1(a)
        out = linear_2(out)
        out.backward()
        adam.step()
        adam.clear_gradients()


class TestMultiTensorAdam(unittest.TestCase):
    def _adam_optimize_dygraph(
        self,
        place,
        use_param_attr=False,
        use_param_group=False,
        use_amp=False,
        use_multi_tensor=False,
    ):
        paddle.disable_static()
        paddle.seed(10)
        paddle.set_device(place)

        input = paddle.randn((5, 5))

        weight_attr = paddle.ParamAttr(
            learning_rate=0.5,
            regularizer=paddle.regularizer.L2Decay(1.0),
            trainable=True,
        )
        if use_param_attr:
            model = paddle.nn.Linear(5, 5, weight_attr=weight_attr)
        else:
            model = paddle.nn.Linear(5, 5)

        if not use_param_group:
            optimizer = paddle.optimizer.Adam(
                parameters=model.parameters(),
                use_multi_tensor=use_multi_tensor,
                multi_precision=use_amp,
            )
        else:
            parameters = list(model.parameters())
            param_num = len(parameters)
            optimizer = paddle.optimizer.Adam(
                parameters=[
                    {
                        'params': parameters[: int(param_num / 2)],
                        'weight_decay': 0.001,
                        'beta1': 0.1,
                        'beta2': 0.99,
                    },
                    {
                        'params': parameters[int(param_num / 2) :],
                        'weight_decay': 0.001,
                        'beta1': 0.1,
                        'beta2': 0.99,
                    },
                ],
                use_multi_tensor=use_multi_tensor,
                multi_precision=use_amp,
            )

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

        return output, model.parameters()

    def _adam_optimize_static(
        self, place, use_amp=False, use_multi_tensor=False
    ):
        paddle.enable_static()
        paddle.seed(10)
        np.random.seed(10)
        if place == 'cpu':
            use_amp = False
        exe = paddle.static.Executor(place=place)
        train_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        optimizer = paddle.optimizer.Adam(
            multi_precision=use_amp, use_multi_tensor=use_multi_tensor
        )

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
                    master_grad=True,
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

    def _get_places(self):
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

    def _check_with_place_amp(self, place, use_amp):
        # test dygraph mode
        output_dygraph1, params_dygraph1 = self._adam_optimize_dygraph(
            place=place, use_amp=use_amp, use_multi_tensor=True
        )
        output_dygraph2, params_dygraph2 = self._adam_optimize_dygraph(
            place=place, use_amp=use_amp, use_multi_tensor=False
        )
        np.testing.assert_allclose(output_dygraph1, output_dygraph2, rtol=1e-05)
        for idx in range(len(params_dygraph1)):
            np.testing.assert_allclose(
                params_dygraph1[idx], params_dygraph2[idx], rtol=1e-05
            )
        with paddle.pir_utils.IrGuard():
            # test static graph mode
            output_static1 = self._adam_optimize_static(
                place=place, use_amp=use_amp, use_multi_tensor=True
            )
            output_static2 = self._adam_optimize_static(
                place=place, use_amp=use_amp, use_multi_tensor=False
            )
            for idx in range(len(output_static1)):
                np.testing.assert_allclose(
                    output_static1[idx], output_static2[idx], rtol=1e-05
                )

    def _check_with_param_arrt(self, place, use_amp):
        output1, params1 = self._adam_optimize_dygraph(
            place=place,
            use_amp=use_amp,
            use_param_attr=True,
            use_multi_tensor=True,
        )
        output2, params2 = self._adam_optimize_dygraph(
            place=place,
            use_amp=use_amp,
            use_param_attr=True,
            use_multi_tensor=False,
        )

        np.testing.assert_allclose(output1, output2, rtol=1e-05)
        for idx in range(len(params1)):
            np.testing.assert_allclose(params1[idx], params2[idx], rtol=1e-05)

    def _check_with_param_group(self, place, use_amp):
        output1, params1 = self._adam_optimize_dygraph(
            place=place,
            use_amp=use_amp,
            use_param_group=True,
            use_multi_tensor=True,
        )
        output2, params2 = self._adam_optimize_dygraph(
            place=place,
            use_amp=use_amp,
            use_param_group=True,
            use_multi_tensor=False,
        )

        np.testing.assert_allclose(output1, output2, rtol=1e-05)
        for idx in range(len(params1)):
            np.testing.assert_allclose(params1[idx], params2[idx], rtol=1e-05)

    def test_main(self):
        for place in self._get_places():
            use_amp_list = [True, False]
            for use_amp in use_amp_list:
                self._check_with_place_amp(place, use_amp)
                self._check_with_param_arrt(place, use_amp)
                self._check_with_param_group(place, use_amp)

    def test_pir_main(self):
        with paddle.pir_utils.IrGuard():
            for place in self._get_places():
                use_amp_list = [True, False]
                for use_amp in use_amp_list:
                    self._check_with_place_amp(place, use_amp)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
