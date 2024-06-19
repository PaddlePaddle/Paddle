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

import paddle
from paddle import base

paddle.enable_static()


class TestFuseBatchNormActPass(unittest.TestCase):
    def build_program(self, main_program, startup_program, use_cuda, seed=1):
        with base.program_guard(main_program, startup_program):
            x = paddle.static.data(
                name='x', shape=[-1, 1, 28, 28], dtype='float32'
            )
            y = paddle.static.data(name="y", shape=[-1, 1], dtype='int64')
            hidden1 = paddle.static.nn.conv2d(
                input=x,
                filter_size=3,
                num_filters=16,
                stride=1,
                padding=1,
                act=None,
                bias_attr=False,
                data_format='NHWC',
            )
            param_attr = base.ParamAttr(
                name='batch_norm_w',
                initializer=paddle.nn.initializer.Constant(value=1.0),
            )
            bias_attr = base.ParamAttr(
                name='batch_norm_b',
                initializer=paddle.nn.initializer.Constant(value=0.0),
            )
            hidden2 = paddle.static.nn.batch_norm(
                input=hidden1,
                param_attr=param_attr,
                bias_attr=bias_attr,
                act='relu',
                data_layout='NHWC',
            )
            hidden3 = paddle.static.nn.fc(x=hidden2, size=32, activation='relu')
            hidden4 = paddle.static.nn.batch_norm(
                input=hidden3, act='relu', data_layout='NHWC'
            )
            prediction = paddle.static.nn.fc(
                x=hidden4, size=10, activation='softmax'
            )
            loss = paddle.nn.functional.cross_entropy(
                input=prediction, label=y, reduction='none', use_softmax=False
            )
            loss = paddle.mean(loss)
            sgd = paddle.optimizer.SGD(learning_rate=0.001)
            if use_cuda:
                sgd = paddle.static.amp.decorate(
                    sgd, use_dynamic_loss_scaling=True, init_loss_scaling=128.0
                )
            sgd.minimize(loss)
        return x, y, loss

    def check(self, place, use_cuda):
        paddle.seed(1)
        paddle.framework.random._manual_program_seed(1)
        main_program = base.Program()
        startup_program = base.Program()
        x, y, loss = self.build_program(main_program, startup_program, use_cuda)
        exe = base.Executor(place)
        iters = 8
        batch_size = 16
        feeder = base.DataFeeder(feed_list=[x, y], place=place)

        # close fused_bn_act_ops
        build_strategy = base.BuildStrategy()
        build_strategy.fuse_bn_act_ops = False
        binary = base.CompiledProgram(
            main_program, build_strategy=build_strategy
        )
        train_reader = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=batch_size
        )
        loss_vals = []
        scope = base.Scope()
        with base.scope_guard(scope):
            exe.run(startup_program)
            for _ in range(iters):
                data = next(train_reader())
                loss_v = exe.run(
                    binary, feed=feeder.feed(data), fetch_list=[loss]
                )
                loss_vals.append(loss_v[0])

        # open fused_bn_act_ops
        build_strategy_fused = base.BuildStrategy()
        build_strategy_fused.fuse_bn_act_ops = True
        binary_fused = base.CompiledProgram(
            main_program, build_strategy=build_strategy_fused
        )
        train_reader_fused = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=batch_size
        )
        loss_vals_fused = []
        scope_fused = base.Scope()
        with base.scope_guard(scope_fused):
            exe.run(startup_program)
            for _ in range(iters):
                data = next(train_reader_fused())
                loss_v = exe.run(
                    binary_fused, feed=feeder.feed(data), fetch_list=[loss]
                )
                loss_vals_fused.append(loss_v[0])

        # check loss
        for i in range(iters):
            self.assertAlmostEqual(loss_vals[i], loss_vals_fused[i], delta=1e-5)

    def test_fuse_bn_act_pass_cpu(self):
        place = base.CPUPlace()
        self.check(place, use_cuda=False)

    def test_fuse_bn_act_pass_cuda(self):
        if base.core.is_compiled_with_cuda():
            place = base.CUDAPlace(0)
            self.check(place, use_cuda=True)


if __name__ == '__main__':
    unittest.main()
