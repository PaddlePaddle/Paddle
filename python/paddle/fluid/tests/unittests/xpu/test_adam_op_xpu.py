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

from __future__ import print_function
import sys
sys.path.append("..")
import unittest
import numpy as np
from op_test import OpTest
from paddle.fluid import core
from paddle.fluid.op import Operator
import paddle.fluid as fluid
import paddle

def adam_step_sparse(inputs, attributes, height, rows, row_numel, np_grad,
                     lazy_mode):
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
        moment1_out[row_id] = beta1 * moment1[row_id] + (1 - beta1
                                                         ) * update_value
        moment2_out[row_id] = beta2 * moment2[row_id] + (
            1 - beta2) * np.square(update_value)
        lr_t = lr * np.sqrt(1 - beta2_pow) / (1 - beta1_pow)
        param_out[row_id] = param[row_id] - lr_t * (moment1_out[row_id] / (
            np.sqrt(moment2_out[row_id]) + epsilon))

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
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 0.001
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
            "LearningRate": np.full((1), 2.0).astype("float32")
        }
        self.init_output = np.full((height, row_numel), 0.0).astype("float32")
        self.attrs = {
            'epsilon': epsilon,
            'beta1': beta1,
            'beta2': beta2,
            'min_row_size_to_use_multithread': 2
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

        param_out, mom1, mom2 = adam_step_sparse(self.dense_inputs, self.attrs,
                                                 height, rows, row_numel,
                                                 np_array, lazy_mode)
        self.outputs = {
            "ParamOut": param_out,
            "Moment1Out": mom1,
            "Moment2Out": mom2,
            'Beta1PowOut': beta1_pow * beta1,
            'Beta2PowOut': beta2_pow * beta2
        }

    def check_with_place(self, place, lazy_mode):
        scope = core.Scope()
        self.setup(scope, place, lazy_mode)

        op_args = dict()
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
            out_var = scope.var(key).get_tensor()
            actual = np.array(out_var)
            actual = actual.reshape([actual.size])
            np_array = np_array.reshape([np_array.size])

            for i in range(np_array.size):
                self.assertLess((actual[i] - np_array[i]), 0.00001)

    def test_sparse_adam(self):
        for lazy_mode in (True, False):
            self.check_with_place(paddle.XPUPlace(0), lazy_mode)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
