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
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op import Operator
from op_test_xpu import XPUOpTest

import paddle
from paddle.base import core


class XPUTestAdamOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'adam'
        self.use_dynamic_create_class = False

    class TestAdamOp(XPUOpTest):
        '''Test Adam Op with supplied attributes'''

        def setUp(self):
            self.init_dtype()
            self.set_xpu()
            self.op_type = "adam"
            self.place = paddle.XPUPlace(0)
            self.set_data()
            self.set_attrs()
            self.set_shape()
            self.set_inputs()
            self.set_steps()
            param_out, moment1_out, moment2_out, moment2_out_max = adam_step(
                self.inputs, self.attrs
            )

            self.outputs = {
                'Moment1Out': moment1_out,
                'Moment2Out': moment2_out,
                'ParamOut': param_out,
                'Beta1PowOut': np.array([self.beta1_pow]).astype("float32")
                * self.beta1,
                'Beta2PowOut': np.array([self.beta2_pow]).astype("float32")
                * self.beta2,
            }

        def set_xpu(self):
            self.__class__.use_xpu = True
            self.__class__.no_need_check_grad = True
            self.__class__.op_type = self.in_type

        def init_dtype(self):
            self.dtype = self.in_type

        def set_attrs(self):
            self.attrs = {
                'epsilon': self.epsilon,
                'beta1': self.beta1,
                'beta2': self.beta2,
            }

        def set_data(self):
            self.beta1 = 0.78
            self.beta2 = 0.836
            self.learning_rate = 0.004
            self.epsilon = 1e-4

        def set_steps(self):
            self.num_steps = 1

        def set_shape(self):
            self.shape = (102, 105)

        def set_inputs(self):
            param = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
            grad = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
            moment1 = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
            # The second moment is positive
            moment2 = np.random.random(self.shape).astype(self.dtype)

            self.beta1_pow = self.beta1**10
            self.beta2_pow = self.beta2**10

            self.inputs = {
                'Param': param,
                'Grad': grad,
                'Moment1': moment1,
                'Moment2': moment2,
                'LearningRate': np.array([self.learning_rate]).astype(
                    "float32"
                ),
                'Beta1Pow': np.array([self.beta1_pow]).astype("float32"),
                'Beta2Pow': np.array([self.beta2_pow]).astype("float32"),
            }

        def test_check_output(self):
            self.check_output_with_place(
                no_check_set=['Moment2MaxOut'],
                place=paddle.XPUPlace(0),
                atol=1e-2,
            )

    class TestAdamOp2(TestAdamOp):
        '''Test Adam Op with supplied attributes'''

        def set_data(self):
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.learning_rate = 0.001
            self.epsilon = 1e-8

    class TestAdamOp3(TestAdamOp2):
        '''Test Adam Op with supplied attributes'''

        def set_shape(self):
            self.shape = (101, 47)

    class TestAdamOp4(TestAdamOp2):
        '''Test Adam Op with supplied attributes'''

        def set_shape(self):
            self.shape = (512, 26)

    class TestAdamOp5(TestAdamOp2):
        '''Test Adam Op with supplied attributes'''

        def set_shape(self):
            self.shape = (11, 1)

    class TestAdamOp6(TestAdamOp2):
        '''Test Adam Op with beta as Variable'''

        def set_shape(self):
            self.shape = (10, 10)

        def set_data(self):
            self.beta1 = 0.85
            self.beta2 = 0.95
            self.learning_rate = 0.001
            self.epsilon = 1e-8

    class TestAdamOp7(TestAdamOp):
        '''Test Adam Op with float16 accuracy'''

        def setUp(self):
            self.init_dtype()
            self.set_xpu()
            self.op_type = "adam"
            self.place = paddle.XPUPlace(0)
            self.set_data()
            self.set_attrs()
            self.set_shape()
            self.set_inputs()
            self.set_steps()
            param_out, moment1_out, moment2_out, moment2_out_max = adam_step(
                self.inputs, self.attrs
            )

            self.outputs = {
                'Moment1Out': moment1_out,
                'Moment2Out': moment2_out,
                'ParamOut': param_out,
                'Beta1PowOut': np.array([self.beta1_pow]).astype("float16")
                * self.beta1,
                'Beta2PowOut': np.array([self.beta2_pow]).astype("float16")
                * self.beta2,
            }

        def set_inputs(self):
            param = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
            grad = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
            moment1 = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
            # The second moment is positive
            moment2 = np.random.random(self.shape).astype(self.dtype)

            self.beta1_pow = self.beta1**10
            self.beta2_pow = self.beta2**10

            self.inputs = {
                'Param': param,
                'Grad': grad,
                'Moment1': moment1,
                'Moment2': moment2,
                'LearningRate': np.array([self.learning_rate]).astype(
                    "float16"
                ),
                'Beta1Pow': np.array([self.beta1_pow]).astype("float16"),
                'Beta2Pow': np.array([self.beta2_pow]).astype("float16"),
            }

    class TestAdamOpMultipleSteps(TestAdamOp2):
        '''Test Adam Operator with supplied attributes'''

        def set_steps(self):
            self.num_steps = 10

        def test_check_output(self):
            for _ in range(self.num_steps):
                param_out, moment1_out, moment2_out, moment2_out_max = (
                    adam_step(self.inputs, self.attrs)
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
                self.check_output_with_place(
                    no_check_set=['Moment2MaxOut'],
                    place=paddle.XPUPlace(0),
                    atol=1e-2,
                )

                # Output of this step becomes input for next step
                self.inputs['Param'] = param_out
                self.inputs['Moment1'] = moment1_out
                self.inputs['Moment2'] = moment2_out

                # Update powers of Beta1 and Beta2 for next time step
                self.inputs['Beta1Pow'] = beta1_pow_out
                self.inputs['Beta2Pow'] = beta2_pow_out

                # Randomize gradient for next step
                self.inputs['Grad'] = np.random.uniform(
                    -1, 1, (102, 105)
                ).astype("float32")


def adam_step(inputs, attributes):
    '''
    Simulate one step of the adam optimizer
    :param inputs: dict of inputs
    :param attributes: dict of attributes
    :return tuple: tuple of output param, moment1, moment2, moment2_max
    beta1 power accumulator and beta2 power accumulator
    '''
    param = inputs['Param']
    grad = inputs['Grad']
    moment1 = inputs['Moment1']
    moment2 = inputs['Moment2']
    moment2_max = inputs.get('Moment2Max', None)
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

    amsgrad = attributes.get('amsgrad', False)

    moment1_out = beta1 * moment1 + (1 - beta1) * grad
    moment2_out = beta2 * moment2 + (1 - beta2) * np.square(grad)

    lr_t = lr * np.sqrt(1 - beta2_pow) / (1 - beta1_pow)

    if amsgrad:
        moment2_max_out = np.maximum(moment2_out, moment2_max)
        param_out = param - lr_t * (
            moment1_out
            / (np.sqrt(moment2_max_out) + epsilon * np.sqrt(1 - beta2_pow))
        )
    else:
        moment2_max_out = np.empty_like(moment2_out)
        param_out = param - lr_t * (
            moment1_out
            / (np.sqrt(moment2_out) + epsilon * np.sqrt(1 - beta2_pow))
        )

    return param_out, moment1_out, moment2_out, moment2_max_out


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
    moment2_max = inputs.get('Moment2Max', None)
    lr = inputs['LearningRate']
    beta1_pow = inputs['Beta1Pow']
    beta2_pow = inputs['Beta2Pow']

    beta1 = attributes['beta1']
    beta2 = attributes['beta2']
    epsilon = attributes['epsilon']
    amsgrad = attributes.get('amsgrad', False)

    moment1_out = np.zeros(shape=[height, row_numel])
    moment2_out = np.zeros(shape=[height, row_numel])
    moment2_max_out = np.zeros(shape=[height, row_numel])
    param_out = np.zeros(shape=[height, row_numel])

    def update_row(row_id, update_value):
        moment1_out[row_id] = (
            beta1 * moment1[row_id] + (1 - beta1) * update_value
        )
        moment2_out[row_id] = beta2 * moment2[row_id] + (1 - beta2) * np.square(
            update_value
        )
        lr_t = lr * np.sqrt(1 - beta2_pow) / (1 - beta1_pow)

        if amsgrad:
            moment2_max_out[row_id] = np.maximum(
                moment2_out[row_id], moment2_max[row_id]
            )
            param_out[row_id] = param[row_id] - lr_t * (
                moment1_out[row_id]
                / (np.sqrt(moment2_max_out[row_id]) + epsilon)
            )
        else:
            moment2_max_out[row_id] = np.empty_like(moment2_out[row_id])
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

    return param_out, moment1_out, moment2_out, moment2_max_out


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
            'amsgrad': False,  # Currently, xpu NOT support amsgrad.
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

        param_out, mom1, mom2, mom2_max = adam_step_sparse(
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

        # create and run adam operator
        adam_op = Operator("adam", **op_args)
        adam_op.run(scope, place)

        for key, np_array in self.outputs.items():
            if key in ['Moment2MaxOut']:  # Currently, xpu NOT support amsgrad.
                continue

            out_var = scope.var(key).get_tensor()
            actual = np.array(out_var)
            actual = actual.reshape([actual.size])
            np_array = np_array.reshape([np_array.size])

            for i in range(np_array.size):
                self.assertLess((actual[i] - np_array[i]), 0.00001)

    def test_sparse_adam(self):
        xpu_version = core.get_xpu_device_version(0)
        version_str = "xpu2" if xpu_version == core.XPUVersion.XPU2 else "xpu1"
        if "xpu2" == version_str:
            self.check_with_place(paddle.XPUPlace(0), False)


class TestSparseAdamOp1(TestSparseAdamOp):
    def setup(self, scope, place, lazy_mode):
        beta1 = 0.78
        beta2 = 0.836
        epsilon = 1e-4
        beta1_pow = np.array([beta1**10]).astype("float16")
        beta2_pow = np.array([beta2**10]).astype("float16")

        height = 10
        rows = [0, 4, 7]
        self.rows = rows
        row_numel = 12
        self.row_numel = row_numel
        self.dense_inputs = {
            "Param": np.full((height, row_numel), 5.0).astype("float16"),
            "Moment1": np.full((height, row_numel), 5.0).astype("float16"),
            "Moment2": np.full((height, row_numel), 5.0).astype("float16"),
            'Beta1Pow': beta1_pow,
            'Beta2Pow': beta2_pow,
            "LearningRate": np.full((1), 2.0).astype("float16"),
        }
        self.init_output = np.full((height, row_numel), 0.0).astype("float16")
        self.attrs = {
            'epsilon': epsilon,
            'beta1': beta1,
            'beta2': beta2,
            'min_row_size_to_use_multithread': 2,
            'amsgrad': False,  # Currently, xpu NOT support amsgrad.
        }

        grad_selected_rows = scope.var('Grad').get_selected_rows()
        grad_selected_rows.set_height(height)
        grad_selected_rows.set_rows(rows)
        np_array = np.ones((len(rows), row_numel)).astype("float16")
        np_array[0, 0] = 2.0
        np_array[2, 8] = 4.0

        grad_tensor = grad_selected_rows.get_tensor()
        grad_tensor.set(np_array, place)

        self.sparse_inputs = ["Grad"]

        param_out, mom1, mom2, mom2_max = adam_step_sparse(
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


support_types = get_xpu_op_support_types('adam')
for stype in support_types:
    create_test_class(globals(), XPUTestAdamOp, stype)

if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
