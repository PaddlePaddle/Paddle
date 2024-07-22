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

import os
import unittest

import numpy as np
from op import Operator
from op_test import OpTest, convert_float_to_uint16

import paddle
from paddle.base import core

paddle.enable_static()


def lamb_wrapper(
    param,
    grad,
    lr,
    moment1,
    moment2,
    beta1Pow,
    beta2Pow,
    master_weight=None,
    found_inf=None,
    epsilon=1e-8,
    beta1=0.9,
    beta2=0.999,
    weight_decay=0.01,
    always_adapt=False,
):
    return paddle._C_ops.lamb_(
        param,
        grad,
        lr,
        moment1,
        moment2,
        beta1Pow,
        beta2Pow,
        master_weight,
        found_inf,
        weight_decay,
        beta1,
        beta2,
        epsilon,
        False,
        False,
    )


class TestLambOp1(OpTest):
    def set_attrs(self):
        self.attrs = {
            'epsilon': 1e-4,
            'beta1': 0.78,
            'beta2': 0.836,
            'weight_decay': 0.01,
            'always_adapt': False,
        }

    def set_dtype(self):
        self.dtype = np.float32

    def setUp(self):
        '''Test Lamb Op with supplied attributes'''
        self.op_type = "lamb"
        self.set_dtype()

        if self.is_bfloat16_op():
            param = np.random.uniform(-1, 1, (102, 105)).astype(np.float32)
            grad = np.random.uniform(-1, 1, (102, 105)).astype(np.float32)
            moment1 = np.random.uniform(-1, 1, (102, 105)).astype(np.float32)
            moment2 = np.random.random((102, 105)).astype(np.float32)
        else:
            param = np.random.uniform(-1, 1, (102, 105)).astype(self.dtype)
            grad = np.random.uniform(-1, 1, (102, 105)).astype(self.dtype)
            moment1 = np.random.uniform(-1, 1, (102, 105)).astype(self.dtype)
            moment2 = np.random.random((102, 105)).astype(self.dtype)

        learning_rate = 0.001
        self.set_attrs()

        self.python_api = lamb_wrapper
        self.python_out_sig = ['Out']

        beta1_pow = self.attrs['beta1']
        beta2_pow = self.attrs['beta2']

        if self.is_bfloat16_op():
            self.inputs = {
                'Param': convert_float_to_uint16(param),
                'Grad': convert_float_to_uint16(grad),
                'Moment1': convert_float_to_uint16(moment1),
                'Moment2': convert_float_to_uint16(moment2),
                'LearningRate': convert_float_to_uint16(
                    np.array([learning_rate]).astype(self.dtype)
                ),
                'Beta1Pow': convert_float_to_uint16(
                    np.array([beta1_pow]).astype(self.dtype)
                ),
                'Beta2Pow': convert_float_to_uint16(
                    np.array([beta2_pow]).astype(self.dtype)
                ),
            }

        else:
            self.inputs = {
                'Param': param,
                'Grad': grad,
                'Moment1': moment1,
                'Moment2': moment2,
                'LearningRate': np.array([learning_rate]).astype(self.dtype),
                'Beta1Pow': np.array([beta1_pow]).astype(self.dtype),
                'Beta2Pow': np.array([beta2_pow]).astype(self.dtype),
            }

        (
            param_out,
            moment1_out,
            moment2_out,
            beta1_pow_out,
            beta2_pow_out,
        ) = lamb_step(self.inputs, self.attrs)

        if self.is_bfloat16_op():
            self.outputs = {
                'Moment1Out': convert_float_to_uint16(moment1_out),
                'Moment2Out': convert_float_to_uint16(moment2_out),
                'ParamOut': convert_float_to_uint16(param_out),
                'Beta1PowOut': convert_float_to_uint16(beta1_pow_out),
                'Beta2PowOut': convert_float_to_uint16(beta2_pow_out),
            }
        else:
            self.outputs = {
                'Moment1Out': moment1_out,
                'Moment2Out': moment2_out,
                'ParamOut': param_out,
                'Beta1PowOut': beta1_pow_out,
                'Beta2PowOut': beta2_pow_out,
            }

    def test_check_output(self):
        self.check_output()


class TestLambOp2(TestLambOp1):
    def set_attrs(self):
        self.attrs = {
            'epsilon': 1e-8,
            'beta1': 0.9,
            'beta2': 0.999,
            'weight_decay': 0.01,
            'always_adapt': False,
        }


class TestLambOp3(TestLambOp1):
    def set_attrs(self):
        self.attrs = {
            'epsilon': 1e-8,
            'beta1': 0.9,
            'beta2': 0.999,
            'weight_decay': 0.0,
            'always_adapt': False,
        }


class TestLambOpMultipleSteps(TestLambOp1):
    def set_attrs(self):
        self.attrs = {
            'epsilon': 1e-8,
            'beta1': 0.9,
            'beta2': 0.999,
            'weight_decay': 0.01,
            'always_adapt': False,
        }
        self.num_steps = 10

    def test_check_output(self):
        for i in range(self.num_steps):
            (
                param_out,
                moment1_out,
                moment2_out,
                beta1_pow_out,
                beta2_pow_out,
            ) = lamb_step(self.inputs, self.attrs)

            self.outputs = {
                'Moment1Out': moment1_out,
                'Moment2Out': moment2_out,
                'ParamOut': param_out,
                'Beta1PowOut': beta1_pow_out,
                'Beta2PowOut': beta2_pow_out,
            }

            # Verify output for this step
            self.check_output()

            # Output of this step becomes input for next step
            self.inputs['Param'] = param_out
            self.inputs['Moment1'] = moment1_out
            self.inputs['Moment2'] = moment2_out

            # Update powers of Beta1 and Beta2 for next time step
            self.inputs['Beta1Pow'] = beta1_pow_out
            self.inputs['Beta2Pow'] = beta2_pow_out

            # Randomize gradient for next step
            self.inputs['Grad'] = np.random.uniform(-1, 1, (102, 105)).astype(
                self.dtype
            )


class TestLambFP16Op1(TestLambOp1):
    def set_dtype(self):
        self.__class__.op_type = "lamb"
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place)


class TestLambFP16Op2(TestLambOp2):
    def set_dtype(self):
        self.__class__.op_type = "lamb"
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place)


class TestLambFP16Op3(TestLambOp3):
    def set_dtype(self):
        self.__class__.op_type = "lamb"
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place)


class TestLambFP16OpMultipleSteps(TestLambOpMultipleSteps):
    def set_dtype(self):
        self.__class__.op_type = "lamb"
        self.dtype = np.float16


class TestLambBF16Op1(TestLambOp1):
    def set_dtype(self):
        self.__class__.op_type = "lamb"
        self.dtype = np.uint16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_bfloat16_supported(place):
                self.check_output_with_place(place)


class TestLambBF16Op2(TestLambOp2):
    def set_dtype(self):
        self.__class__.op_type = "lamb"
        self.dtype = np.uint16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_bfloat16_supported(place):
                self.check_output_with_place(place)


class TestLambBF16Op3(TestLambOp3):
    def set_dtype(self):
        self.__class__.op_type = "lamb"
        self.dtype = np.uint16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_bfloat16_supported(place):
                self.check_output_with_place(place)


class TestLambBF16OpMultipleSteps(TestLambOpMultipleSteps):
    def set_dtype(self):
        self.__class__.op_type = "lamb"
        self.dtype = np.uint16

    def test_check_output(self):
        for i in range(self.num_steps):
            (
                param_out,
                moment1_out,
                moment2_out,
                beta1_pow_out,
                beta2_pow_out,
            ) = lamb_step(self.inputs, self.attrs)

            self.outputs = {
                'Moment1Out': convert_float_to_uint16(moment1_out),
                'Moment2Out': convert_float_to_uint16(moment2_out),
                'ParamOut': convert_float_to_uint16(param_out),
                'Beta1PowOut': convert_float_to_uint16(beta1_pow_out),
                'Beta2PowOut': convert_float_to_uint16(beta2_pow_out),
            }

            # Verify output for this step
            if core.is_compiled_with_cuda():
                place = core.CUDAPlace(0)
                if core.is_bfloat16_supported(place):
                    self.check_output_with_place(place)

            # Output of this step becomes input for next step
            self.inputs['Param'] = convert_float_to_uint16(param_out)
            self.inputs['Moment1'] = convert_float_to_uint16(moment1_out)
            self.inputs['Moment2'] = convert_float_to_uint16(moment2_out)

            # Update powers of Beta1 and Beta2 for next time step
            self.inputs['Beta1Pow'] = convert_float_to_uint16(beta1_pow_out)
            self.inputs['Beta2Pow'] = convert_float_to_uint16(beta2_pow_out)

            # Randomize gradient for next step
            self.inputs['Grad'] = convert_float_to_uint16(
                np.random.uniform(-1, 1, (102, 105)).astype(np.float32)
            )


def lamb_step(inputs, attributes):
    '''
    Simulate one step of the lamb optimizer
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

    beta1 = attributes['beta1']
    beta2 = attributes['beta2']
    epsilon = attributes['epsilon']
    weight_decay = attributes['weight_decay']
    always_adapt = attributes['always_adapt']

    moment1_out = beta1 * moment1 + (1 - beta1) * grad
    moment2_out = beta2 * moment2 + (1 - beta2) * np.square(grad)

    moment1_unbiased = moment1_out / (1 - beta1_pow)
    moment2_unbiased = moment2_out / (1 - beta2_pow)

    if weight_decay > 0 or always_adapt:
        r_1 = np.linalg.norm(param)
        r_2 = np.linalg.norm(
            moment1_unbiased / (np.sqrt(moment2_unbiased) + epsilon)
            + weight_decay * param
        )
        lr_t = lr * r_1 / r_2
    else:
        lr_t = lr

    param_out = param - lr_t * (
        moment1_unbiased / (np.sqrt(moment2_unbiased) + epsilon)
        + weight_decay * param
    )

    beta1_pow_out = beta1_pow * beta1
    beta2_pow_out = beta2_pow * beta2

    return param_out, moment1_out, moment2_out, beta1_pow_out, beta2_pow_out


def lamb_step_sparse(inputs, attributes, height, rows, row_numel, np_grad):
    '''
    Simulate one step of the lamb optimizer
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
    weight_decay = attributes['weight_decay']
    always_adapt = attributes['always_adapt']

    moment1_out = np.zeros(shape=[height, row_numel])
    moment2_out = np.zeros(shape=[height, row_numel])
    param_out = np.zeros(shape=[height, row_numel])
    moment1_unbiased = np.zeros(shape=[height, row_numel])
    moment2_unbiased = np.zeros(shape=[height, row_numel])

    def update_mom(row_id, update_value):
        moment1_out[row_id] = (
            beta1 * moment1[row_id] + (1 - beta1) * update_value
        )
        moment2_out[row_id] = beta2 * moment2[row_id] + (1 - beta2) * np.square(
            update_value
        )

        moment1_out[row_id] = (
            beta1 * moment1[row_id] + (1 - beta1) * update_value
        )
        moment2_out[row_id] = beta2 * moment2[row_id] + (1 - beta2) * np.square(
            update_value
        )

    def update_param(weight_decay, always_adapt):
        if weight_decay > 0 or always_adapt:
            r_1 = np.linalg.norm(param)
            r_2 = np.linalg.norm(
                moment1_out / (np.sqrt(moment2_out) + epsilon)
                + weight_decay * param
            )
            lr_t = lr * r_1 / r_2
        else:
            lr_t = lr

        param_out = param - lr_t * (
            moment1_out / (np.sqrt(moment2_out) + epsilon)
            + weight_decay * param
        )

    for row_id in range(param_out.shape[0]):
        update_value = np.zeros(np_grad[0].shape).astype("float32")
        if row_id in rows:
            update_value = np_grad[rows.index(row_id)]
        update_mom(row_id, update_value)

    update_param(weight_decay, always_adapt)
    beta1_pow_out = beta1_pow * beta1
    beta2_pow_out = beta2_pow * beta2

    return param_out, moment1_out, moment2_out, beta1_pow_out, beta2_pow_out


class TestSparseLambOp(unittest.TestCase):
    def setup(self, scope, place):
        beta1 = 0.78
        beta2 = 0.836
        epsilon = 1e-4

        height = 10
        rows = [0, 4, 7]
        self.rows = rows
        row_numel = 12
        self.row_numel = row_numel
        self.dense_inputs = {
            "Param": np.full((height, row_numel), 5.0).astype("float32"),
            "Moment1": np.full((height, row_numel), 5.0).astype("float32"),
            "Moment2": np.full((height, row_numel), 5.0).astype("float32"),
            'Beta1Pow': np.array([beta1]).astype("float32"),
            'Beta2Pow': np.array([beta2]).astype("float32"),
            "LearningRate": np.full((1), 2.0).astype("float32"),
        }
        self.init_output = np.full((height, row_numel), 0.0).astype("float32")
        self.attrs = {
            'epsilon': epsilon,
            'beta1': beta1,
            'beta2': beta2,
            'weight_decay': 0.05,
            'always_adapt': False,
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

        param_out, mom1, mom2, beta1_pow_out, beta2_pow_out = lamb_step_sparse(
            self.dense_inputs, self.attrs, height, rows, row_numel, np_array
        )
        self.outputs = {
            "ParamOut": param_out,
            "Moment1Out": mom1,
            "Moment2Out": mom2,
            'Beta1PowOut': beta1_pow_out,
            'Beta2PowOut': beta2_pow_out,
        }

    def check_with_place(self, place):
        scope = core.Scope()
        self.setup(scope, place)

        op_args = {}
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
        lamb_op = Operator("lamb", **op_args)
        lamb_op.run(scope, place)

        for key, np_array in self.outputs.items():
            out_var = scope.var(key).get_tensor()
            actual = np.array(out_var)
            actual = actual.reshape([actual.size])
            np_array = np_array.reshape([np_array.size])

            for i in range(np_array.size):
                self.assertLess((actual[i] - np_array[i]), 0.00001)

    def test_sparse_lamb(self):
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


if __name__ == "__main__":
    unittest.main()
