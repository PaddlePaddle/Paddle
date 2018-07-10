# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.framework as framework
import numpy as np
from op_test import OpTest
from op_builder import OpBuilder


class BaseTestCase(OpTest):
    def initTestCase(self):
        self.shape = (4, 3, 224, 224)
        self.dtype = 'float16'
        self.dropout_prob = 0.2
        self.is_test = False
        self.seed = 1

        self.num_tol = 1e-6
        self.dropout_prob_tol = 1e-2

    def setUp(self):
        self.initTestCase()

        self.op_type = 'dropout'
        self.fix_seed = self.seed is not None

        if self.dropout_prob in [0.0, 1.0]:
            self.dropout_prob_tol = 0

        self.x = np.random.uniform(
            low=1, high=10, size=self.shape).astype(self.dtype)

        self.grad_out = np.random.uniform(
            low=1, high=10, size=self.shape).astype(self.dtype)

        self.inputs = {'X': self.x}
        self.outputs = {'Out': None, 'States': None, 'ReserveSpace': None}
        self.attrs = {
            'is_test': self.is_test,
            'seed': self.seed,
            'fix_seed': self.fix_seed,
            'dropout_prob': float(self.dropout_prob),
            'use_cudnn': True
        }

        self.place = fluid.CUDAPlace(0)

    def test(self):
        mask1 = self.base_check()
        if self.seed is not None:
            mask2 = self.base_check()
            self.assertTrue(np.array_equal(mask1, mask2), 'Fix seed error')

    def base_check(self):
        fwd_mask = self.fwd_exec()
        if not self.is_test:
            bwd_mask = self.bwd_exec()
            self.assertTrue(
                np.array_equal(fwd_mask, bwd_mask),
                'Forward mask and backward mask not match')
        return fwd_mask

    def fwd_exec(self):
        self.scope = core.Scope()
        fwd_builder = OpBuilder(self.scope, self.place)
        fwd_builder.set_type(self.op_type)

        for name, value in self.inputs.items():
            fwd_builder.add_input(name, value=value)

        for name, value in self.outputs.items():
            fwd_builder.add_output(name)

        for name, value in self.attrs.items():
            fwd_builder.add_attr(name, value=value)

        y, = fwd_builder.build_and_run(fetch_list=['Out'])
        return self.validate_and_return_mask(self.x, y)

    def bwd_exec(self):
        new_scope = core.Scope()
        bwd_builder = OpBuilder(new_scope, self.place)
        bwd_builder.set_type(self.op_type + '_grad')
        for name, value in self.outputs.items():
            if name == 'Out':
                bwd_builder.add_input(
                    framework.grad_var_name(name), value=self.grad_out)
            else:
                value = np.array(self.scope.var(name).get_tensor())
                bwd_builder.add_input(name, value=value)

        for name, value in self.inputs.items():
            bwd_builder.add_output(framework.grad_var_name(name))

        for name, value in self.attrs.items():
            bwd_builder.add_attr(name, value=value)

        grad_x, = bwd_builder.build_and_run(
            fetch_list=[framework.grad_var_name('X')])

        return self.validate_and_return_mask(self.grad_out, grad_x)

    def validate_and_return_mask(self, x, y):
        self.assertEqual(x.shape, y.shape)
        mask1 = (abs(x - y) <= self.num_tol)
        mask2 = (y != 0)
        self.assertTrue(
            np.array_equal(mask1, mask2), "Dropout op arithmetic error")
        dropout_num = x.size - mask1.sum()
        actual_dropout_ratio = float(dropout_num) / x.size
        self.assertTrue(
            abs(actual_dropout_ratio - self.dropout_prob) <=
            self.dropout_prob_tol,
            "Set dropout_prob={} but actual is {}".format(self.dropout_prob,
                                                          actual_dropout_ratio))
        return mask1


if __name__ == '__main__':
    unittest.main()
