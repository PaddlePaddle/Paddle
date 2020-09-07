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

import unittest
import numpy as np
from op_test import OpTest
import paddle
import paddle.fluid as fluid
from paddle.fluid import core


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "Paddle core is not compiled with CUDA")
class TestFusedBnAddActAPI(unittest.TestCase):
    def build_fused_program(self,
                            main_program,
                            startup_program,
                            use_cuda,
                            seed=1):
        with fluid.program_guard(main_program, startup_program):
            x = fluid.layers.data(name='x', shape=[1, 28, 28], dtype='float32')
            y = fluid.layers.data(name="y", shape=[1], dtype='int64')
            hidden1_1 = fluid.layers.conv2d(
                input=x,
                filter_size=3,
                num_filters=32,
                stride=1,
                padding=1,
                act=None,
                bias_attr=False,
                data_format='NHWC')  #[-1, 1, 28, 32]
            hidden1_2 = fluid.layers.conv2d(
                input=x,
                filter_size=3,
                num_filters=32,
                stride=1,
                padding=1,
                act=None,
                bias_attr=False,
                data_format='NHWC')
            param_attr = fluid.ParamAttr(
                name='batch_norm_w',
                initializer=fluid.initializer.Constant(value=1.0))
            bias_attr = fluid.ParamAttr(
                name='batch_norm_b',
                initializer=fluid.initializer.Constant(value=0.0))
            hidden2 = fluid.layers.batch_norm(
                input=hidden1_1,
                param_attr=param_attr,
                bias_attr=bias_attr,
                act='relu',
                data_layout='NHWC')
            hidden1_2 = fluid.layers.cast(hidden1_2, dtype="float16")
            hidden2 = fluid.layers.cast(hidden2, dtype="float16")
            hidden3 = fluid.contrib.layers.fused_bn_add_act(hidden1_2, hidden2)
            prediction = fluid.layers.fc(input=hidden3, size=10, act='softmax')
            loss = fluid.layers.cross_entropy(input=prediction, label=y)
            loss = fluid.layers.mean(loss)
            sgd = fluid.optimizer.SGD(learning_rate=0.001)
            sgd.minimize(loss)

        return x, y, loss

    def build_origin_program(self,
                             main_program,
                             startup_program,
                             use_cuda,
                             seed=1):
        with fluid.program_guard(main_program, startup_program):
            x = fluid.layers.data(name='x', shape=[1, 28, 28], dtype='float32')
            y = fluid.layers.data(name="y", shape=[1], dtype='int64')
            hidden1_1 = fluid.layers.conv2d(
                input=x,
                filter_size=3,
                num_filters=32,
                stride=1,
                padding=1,
                act=None,
                bias_attr=False,
                data_format='NHWC')
            hidden1_2 = fluid.layers.conv2d(
                input=x,
                filter_size=3,
                num_filters=32,
                stride=1,
                padding=1,
                act=None,
                bias_attr=False,
                data_format='NHWC')
            param_attr = fluid.ParamAttr(
                name='batch_norm_w',
                initializer=fluid.initializer.Constant(value=1.0))
            bias_attr = fluid.ParamAttr(
                name='batch_norm_b',
                initializer=fluid.initializer.Constant(value=0.0))
            hidden2 = fluid.layers.batch_norm(
                input=hidden1_1,
                param_attr=param_attr,
                bias_attr=bias_attr,
                act='relu',
                data_layout='NHWC')
            hidden1_2 = fluid.layers.cast(hidden1_2, dtype="float16")
            hidden2 = fluid.layers.cast(hidden2, dtype="float16")
            hidden3 = fluid.layers.batch_norm(hidden1_2)
            hidden3 = hidden3 + hidden2
            hidden3 = fluid.layers.relu(hidden3)
            prediction = fluid.layers.fc(input=hidden3, size=10, act='softmax')
            loss = fluid.layers.cross_entropy(input=prediction, label=y)
            loss = fluid.layers.mean(loss)
            sgd = fluid.optimizer.SGD(learning_rate=0.001)
            sgd.minimize(loss)

        return x, y, loss

    def check(self, place, use_cuda):
        paddle.manual_seed(1)
        paddle.framework.random._manual_program_seed(1)
        iters = 2
        batch_size = 1

        # build_fused_program
        main_program = fluid.Program()
        startup_program = fluid.Program()
        x, y, loss = self.build_fused_program(main_program, startup_program,
                                              use_cuda)
        exe = fluid.Executor(place)
        feeder = fluid.DataFeeder(feed_list=[x, y], place=place)
        #binary = fluid.CompiledProgram(main_program).with_data_parallel(
        #    loss_name=loss.name)
        train_reader = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=batch_size)
        loss_vals_fused = []
        scope = fluid.Scope()
        with fluid.scope_guard(scope):
            exe.run(startup_program)
            for _ in range(iters):
                data = next(train_reader())
                loss_v = exe.run(main_program,
                                 feed=feeder.feed(data),
                                 fetch_list=[loss])
                loss_vals_fused.append(loss_v[0][0])

        # build_origin_program
        main_program = fluid.Program()
        startup_program = fluid.Program()
        x, y, loss = self.build_origin_program(main_program, startup_program,
                                               use_cuda)
        feeder = fluid.DataFeeder(feed_list=[x, y], place=place)
        #binary = fluid.CompiledProgram(main_program).with_data_parallel(
        #    loss_name=loss.name, build_strategy=build_strategy)
        train_reader = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=batch_size)
        loss_vals = []
        scope = fluid.Scope()
        with fluid.scope_guard(scope):
            exe.run(startup_program)
            for _ in range(iters):
                data = next(train_reader())
                loss_v = exe.run(main_program,
                                 feed=feeder.feed(data),
                                 fetch_list=[loss])
                loss_vals.append(loss_v[0][0])

        # check loss
        for i in range(iters):
            self.assertAlmostEqual(loss_vals[i], loss_vals_fused[i], delta=1e-5)

    def test_fuse_bn_add_act(self):
        place = fluid.CUDAPlace(0)
        self.check(place, use_cuda=True)


if __name__ == '__main__':
    unittest.main()
