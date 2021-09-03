#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import random
import sys
import unittest
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.layers as layers
import paddle.nn as nn
from op_test import OpTest

paddle.enable_static()


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "Paddle core is not compiled with CUDA")
class TestResNetUnitAPI(unittest.TestCase):
    def setUp(self):
        self.conv_param_attr1 = fluid.ParamAttr(
            name='conv2d_1.weight',
            initializer=fluid.initializer.Xavier(uniform=False),
            learning_rate=0.001)
        self.conv_param_attr2 = fluid.ParamAttr(
            name='conv2d_2.weight',
            initializer=fluid.initializer.Xavier(uniform=False),
            learning_rate=0.001)
        self.bn_param_attr1 = fluid.ParamAttr(
            name='batch_norm_w_1',
            initializer=fluid.initializer.Constant(value=1.0))
        self.bn_bias_attr1 = fluid.ParamAttr(
            name='batch_norm_b_1',
            initializer=fluid.initializer.Constant(value=0.0))
        self.bn_param_attr2 = fluid.ParamAttr(
            name='batch_norm_w_2',
            initializer=fluid.initializer.Constant(value=1.0))
        self.bn_bias_attr2 = fluid.ParamAttr(
            name='batch_norm_b_2',
            initializer=fluid.initializer.Constant(value=0.0))
        self.fc_param_attr = fluid.ParamAttr(
            name='fc.weight',
            initializer=fluid.initializer.Xavier(uniform=False))

    def build_fused_program(self,
                            main_program,
                            startup_program,
                            use_cuda,
                            seed=1):
        with fluid.program_guard(main_program, startup_program):
            x = fluid.layers.data(name='x', shape=[2, 8, 8, 8], dtype='float16')
            y = fluid.layers.data(
                name="y", shape=[2, 32, 8, 8], dtype='float16')
            z = x
            resnet_unit = nn.ResNetUnit(
                num_channels=8,
                num_filters=32,
                filter_size=1,
                stride=1,
                ele_count=128,
                fused_add=True,
                has_shortcut=True,
                filter_x_attr=self.conv_param_attr1,
                scale_x_attr=self.bn_param_attr1,
                bias_x_attr=self.bn_bias_attr1,
                filter_z_attr=self.conv_param_attr2,
                scale_z_attr=self.bn_param_attr2,
                bias_z_attr=self.bn_bias_attr2)
            prediction = resnet_unit(x, z)
            loss = fluid.layers.cross_entropy(input=prediction, label=y)
            loss = fluid.layers.mean(loss)
            sgd = fluid.optimizer.SGD(learning_rate=0.001)
            sgd = fluid.contrib.mixed_precision.decorate(
                sgd, use_dynamic_loss_scaling=True, init_loss_scaling=128.0)
            sgd.minimize(loss)

        return loss

    def build_origin_program(self,
                             main_program,
                             startup_program,
                             use_cuda,
                             seed=1):
        with fluid.program_guard(main_program, startup_program):
            x = fluid.layers.data(name='x', shape=[2, 8, 8, 8], dtype='float16')
            y = fluid.layers.data(
                name="y", shape=[2, 32, 8, 8], dtype='float16')
            conv1_1 = fluid.layers.conv2d(
                input=x,
                filter_size=1,
                num_filters=32,
                stride=1,
                padding=0,
                act=None,
                param_attr=self.conv_param_attr1,
                bias_attr=False,
                data_format='NHWC')
            bn1 = fluid.layers.batch_norm(
                input=conv1_1,
                param_attr=self.bn_param_attr1,
                bias_attr=self.bn_bias_attr1,
                act=None,
                data_layout='NHWC')
            conv1_2 = fluid.layers.conv2d(
                input=x,
                filter_size=1,
                num_filters=32,
                stride=1,
                padding=0,
                act=None,
                param_attr=self.conv_param_attr2,
                bias_attr=False,
                data_format='NHWC')
            bn2 = fluid.layers.batch_norm(
                input=conv1_2,
                param_attr=self.bn_param_attr2,
                bias_attr=self.bn_bias_attr2,
                act=None,
                data_layout='NHWC')
            out = bn1 + bn2
            prediction = fluid.layers.relu(out)
            loss = fluid.layers.cross_entropy(input=prediction, label=y)
            loss = fluid.layers.mean(loss)
            sgd = fluid.optimizer.SGD(learning_rate=0.001)
            sgd = fluid.contrib.mixed_precision.decorate(
                sgd, use_dynamic_loss_scaling=True, init_loss_scaling=128.0)
            sgd.minimize(loss)

        return loss

    def check(self, place, use_cuda):
        paddle.seed(1)
        paddle.framework.random._manual_program_seed(1)
        iters = 5

        exe = fluid.Executor(place)
        # build_fused_program: turn on resnet_unit_op
        main_program = fluid.Program()
        startup_program = fluid.Program()
        loss = self.build_fused_program(main_program, startup_program, use_cuda)
        build_strategy_fused = fluid.BuildStrategy()
        fused = fluid.CompiledProgram(main_program).with_data_parallel(
            loss_name=loss.name, build_strategy=build_strategy_fused)
        loss_vals_fused = []
        x_data = []
        y_data = []
        scope = fluid.Scope()
        with fluid.scope_guard(scope):
            exe.run(startup_program)
            for _ in range(iters):
                x = np.random.random((2, 8, 8, 8)).astype("float16")
                y = np.random.random((2, 32, 8, 8)).astype("float16")
                x_data.append(x)
                y_data.append(y)
                loss_v = exe.run(fused,
                                 feed={"x": x,
                                       "y": y},
                                 fetch_list=[loss])
                loss_vals_fused.append(loss_v[0][0])

        # build_origin_program: turn off resnet_unit_op
        build_strategy = fluid.BuildStrategy()
        origin = fluid.CompiledProgram(main_program).with_data_parallel(
            loss_name=loss.name, build_strategy=build_strategy_fused)
        loss_vals = []
        scope = fluid.Scope()
        with fluid.scope_guard(scope):
            exe.run(startup_program)
            for i in range(iters):
                loss_v = exe.run(origin,
                                 feed={"x": x_data[i],
                                       "y": y_data[i]},
                                 fetch_list=[loss])
                loss_vals.append(loss_v[0][0])

        # check loss
        for i in range(iters):
            self.assertAlmostEqual(loss_vals[i], loss_vals_fused[i], delta=1e-5)

    def test_resnet_unit(self):
        place = fluid.CUDAPlace(0)
        self.check(place, use_cuda=True)


if __name__ == '__main__':
    unittest.main()
