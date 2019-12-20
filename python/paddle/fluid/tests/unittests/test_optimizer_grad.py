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

from __future__ import print_function

import unittest

import paddle.fluid as fluid
import paddle.fluid.optimizer as optimizer
import numpy as np
from paddle.fluid.backward import _append_grad_suffix_

np.random.seed(10)


class SimpleNetWithCond(object):
    def __init__(self, optimizer, param_lr=1.0, y_no_grad=False):
        self.optimizer = optimizer
        self.param_lr = param_lr
        self.shape = [5, 10]
        self.y_no_grad = y_no_grad
        self._init_param()

    def _init_param(self):
        self.x = np.random.random(self.shape).astype('float32')
        self.y = np.random.random(self.shape).astype('float32')
        self.z = np.random.random(self.shape).astype('float32')

    def _calc_gradient(self, cond_i):

        d_out_val = 1. / np.prod(self.shape)
        x_grad = np.ones_like(self.x) * d_out_val
        if cond_i > 1:
            y_grad_ratio, z_grad_ratio = 0 if self.y_no_grad else 3, 1
        else:
            y_grad_ratio, z_grad_ratio = 3, 0
        y_grad = np.ones_like(self.y) * d_out_val * y_grad_ratio
        z_grad = np.ones_like(self.z) * d_out_val * z_grad_ratio

        param_lr = self.param_lr * self.optimizer._learning_rate
        x_new = self.x - param_lr * x_grad
        y_new = self.y - param_lr * y_grad
        z_new = self.z - param_lr * z_grad

        return [x_new, z_new, x_grad, z_grad] if self.y_no_grad else [
            x_new, y_new, z_new, x_grad, y_grad, z_grad
        ]

    def build_net(self, cond_i):
        """
        pseudo code:
            sum_xy = x + y
            sub_yz = y - z
            if i > 1:
                internal = y + z
                sum_cond = internal + z
            else:
                sum_cond = y + z
            sum_all = sum_xy + sum_yz + sum_cond
            mean_out = mean(sum_all)
            optimizer.minimize(mean_out)
        """
        param_x = fluid.layers.create_parameter(
            dtype="float32",
            shape=self.shape,
            attr=fluid.ParamAttr(
                learning_rate=self.param_lr, name="param_x"),
            default_initializer=fluid.initializer.NumpyArrayInitializer(self.x))

        param_y = fluid.layers.create_parameter(
            dtype="float32",
            shape=self.shape,
            attr=fluid.ParamAttr(
                learning_rate=self.param_lr, name="param_y"),
            default_initializer=fluid.initializer.NumpyArrayInitializer(self.y))
        param_z = fluid.layers.create_parameter(
            dtype="float32",
            shape=self.shape,
            attr=fluid.ParamAttr(
                learning_rate=self.param_lr, name="param_z"),
            default_initializer=fluid.initializer.NumpyArrayInitializer(self.z))

        sum_xy = fluid.layers.elementwise_add(param_x, param_y, name='sum_xy')
        sub_yz = fluid.layers.elementwise_sub(param_y, param_z, name='sub_yz')
        useless = fluid.layers.fc(param_x, size=1, name='fc_useless')

        def cond_true():
            interal = fluid.layers.elementwise_add(
                param_y, param_z, name='sum_inter')
            # param_y will not be updated
            param_y.stop_gradient = self.y_no_grad
            cond_res = fluid.layers.elementwise_add(
                interal, param_z, name='sum_cond_true')
            cond_useless = fluid.layers.elementwise_mul(param_x, param_y)
            return cond_res

        def cond_false():
            cond_res = fluid.layers.elementwise_add(
                param_y, param_z, name='sum_cond_false')
            cond_useless = fluid.layers.elementwise_mul(param_z, param_z)
            return cond_res

        cond_i = fluid.layers.assign(np.array([cond_i], dtype='float32'))
        sum_cond = fluid.layers.cond(cond_i > 1.0, cond_true, cond_false)
        sum_all = fluid.layers.sum([sum_xy, sub_yz, sum_cond])
        mean_out = fluid.layers.mean(sum_all)
        opts, params_grads = self.optimizer.minimize(mean_out)

        fetch_list = ["param_x", "param_z"] if self.y_no_grad else [
            "param_x", "param_y", "param_z"
        ]
        fetch_list += [_append_grad_suffix_(param) for param in fetch_list]
        return opts, fetch_list


class TestOptimizer(unittest.TestCase):
    def setUp(self):
        self.optimizer = optimizer.SGDOptimizer(learning_rate=1.)
        self.NetClass = SimpleNetWithCond
        self.param_lr = [1.0, 2.0]
        self.cond_i = [0.1, 3]
        self.y_no_grad = [True, False]
        self.opts_scale = ["scale", "sgd"]
        self.opts_no_scale = ["sgd"]

    def test_optimizer(self):
        self._check_opts()
        self._check_grads()

    def _check_opts(self):
        """
        To check the validity returned opts.
        """
        # if param_lr not set 1.0, it will create a scale_op to scale lr.
        for param_lr in self.param_lr:
            for cond_i in self.cond_i:
                for y_no_grad in self.y_no_grad:
                    main_program = fluid.Program()
                    with fluid.program_guard(main_program):
                        test_net = self.NetClass(
                            self.optimizer,
                            param_lr=param_lr,
                            y_no_grad=y_no_grad)
                        opts, _ = test_net.build_net(cond_i)
                        self.assertListEqual([op.type for op in opts],
                                             self._get_opts(param_lr,
                                                            y_no_grad))

    def _check_grads(self):
        """
        To check the validity of apply_optimize.
        """
        for param_lr in self.param_lr:
            for cond_i in self.cond_i:
                for y_no_grad in self.y_no_grad:
                    main_program = fluid.Program()
                    init_program = fluid.Program()
                    with fluid.program_guard(main_program, init_program):
                        test_net = SimpleNetWithCond(
                            self.optimizer,
                            param_lr=param_lr,
                            y_no_grad=y_no_grad)
                        test_net._init_param()
                        opts, fetch_list = test_net.build_net(cond_i)
                        place = fluid.CPUPlace()
                        exe = fluid.Executor(place)
                        exe.run(init_program)
                        res = exe.run(main_program,
                                      feed={},
                                      fetch_list=fetch_list)

                        gt_grad = test_net._calc_gradient(cond_i)
                        for i in range(len(fetch_list)):
                            np.testing.assert_equal(res[i], gt_grad[i])

    def _get_opts(self, param_lr, y_no_grad):
        """
        Return the optimize_ops of different net
        """
        if param_lr != 1.0:
            if y_no_grad:
                return self.opts_scale * 2
            else:
                return self.opts_scale * 3
        else:
            if y_no_grad:
                return self.opts_no_scale * 2
            else:
                return self.opts_no_scale * 3


if __name__ == '__main__':
    unittest.main()
