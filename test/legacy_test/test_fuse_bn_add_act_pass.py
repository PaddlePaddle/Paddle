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

import unittest

import numpy as np

import paddle
import paddle.nn.functional as F
from paddle import base
from paddle.base import core

paddle.enable_static()


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "Paddle core is not compiled with CUDA"
)
class TestFusedBnAddActAPI(unittest.TestCase):
    def setUp(self):
        self.conv_param_attr1 = base.ParamAttr(
            name='conv2d_1.weight',
            initializer=paddle.nn.initializer.XavierNormal(),
            learning_rate=0.001,
        )
        self.conv_param_attr2 = base.ParamAttr(
            name='conv2d_2.weight',
            initializer=paddle.nn.initializer.XavierNormal(),
            learning_rate=0.001,
        )
        self.bn_param_attr1 = base.ParamAttr(
            name='batch_norm_w_1',
            initializer=paddle.nn.initializer.Constant(value=1.0),
        )
        self.bn_bias_attr1 = base.ParamAttr(
            name='batch_norm_b_1',
            initializer=paddle.nn.initializer.Constant(value=0.0),
        )
        self.bn_param_attr2 = base.ParamAttr(
            name='batch_norm_w_2',
            initializer=paddle.nn.initializer.Constant(value=1.0),
        )
        self.bn_bias_attr2 = base.ParamAttr(
            name='batch_norm_b_2',
            initializer=paddle.nn.initializer.Constant(value=0.0),
        )
        self.fc_param_attr = base.ParamAttr(
            name='fc.weight',
            initializer=paddle.nn.initializer.XavierNormal(),
        )

    def build_fused_program(
        self, main_program, startup_program, use_cuda, seed=1
    ):
        with base.program_guard(main_program, startup_program):
            x = paddle.static.data(
                name='x', shape=[-1, 1, 28, 28], dtype='float32'
            )
            y = paddle.static.data(name="y", shape=[-1, 1], dtype='int64')
            conv1_1 = paddle.static.nn.conv2d(
                input=x,
                filter_size=3,
                num_filters=32,
                stride=1,
                padding=1,
                act=None,
                param_attr=self.conv_param_attr1,
                bias_attr=False,
                data_format='NHWC',
            )
            conv1_2 = paddle.static.nn.conv2d(
                input=x,
                filter_size=3,
                num_filters=32,
                stride=1,
                padding=1,
                act=None,
                param_attr=self.conv_param_attr2,
                bias_attr=False,
                data_format='NHWC',
            )
            bn = paddle.static.nn.batch_norm(
                input=conv1_1,
                param_attr=self.bn_param_attr1,
                bias_attr=self.bn_bias_attr1,
                act=None,
                data_layout='NHWC',
            )
            fused_bn_add_act = paddle.incubate.layers.nn.fused_bn_add_act(
                conv1_2,
                bn,
                param_attr=self.bn_param_attr2,
                bias_attr=self.bn_bias_attr2,
            )
            prediction = paddle.static.nn.fc(
                x=fused_bn_add_act,
                size=10,
                activation='softmax',
                weight_attr=self.fc_param_attr,
            )
            loss = paddle.nn.functional.cross_entropy(
                input=prediction, label=y, reduction='none', use_softmax=False
            )
            loss = paddle.mean(loss)
            sgd = paddle.optimizer.SGD(learning_rate=0.001)
            sgd = paddle.static.amp.decorate(
                sgd, use_dynamic_loss_scaling=True, init_loss_scaling=128.0
            )
            sgd.minimize(loss)

        return x, y, loss

    def build_origin_program(
        self, main_program, startup_program, use_cuda, seed=1
    ):
        with base.program_guard(main_program, startup_program):
            x = paddle.static.data(
                name='x', shape=[-1, 1, 28, 28], dtype='float32'
            )
            y = paddle.static.data(name="y", shape=[-1, 1], dtype='int64')
            conv1_1 = paddle.static.nn.conv2d(
                input=x,
                filter_size=3,
                num_filters=32,
                stride=1,
                padding=1,
                act=None,
                param_attr=self.conv_param_attr1,
                bias_attr=False,
                data_format='NHWC',
            )
            bn1 = paddle.static.nn.batch_norm(
                input=conv1_1,
                param_attr=self.bn_param_attr1,
                bias_attr=self.bn_bias_attr1,
                act=None,
                data_layout='NHWC',
            )
            conv1_2 = paddle.static.nn.conv2d(
                input=conv1_1,
                filter_size=1,
                num_filters=32,
                stride=1,
                act=None,
                param_attr=self.conv_param_attr2,
                bias_attr=False,
                data_format='NHWC',
            )
            bn2 = paddle.static.nn.batch_norm(
                input=conv1_1,
                param_attr=self.bn_param_attr2,
                bias_attr=self.bn_bias_attr2,
                act=None,
                data_layout='NHWC',
            )
            out = bn1 + bn2
            out = F.relu(out)
            prediction = paddle.static.nn.fc(
                x=out,
                size=10,
                activation='softmax',
                weight_attr=self.fc_param_attr,
            )
            loss = paddle.nn.functional.cross_entropy(
                input=prediction, label=y, reduction='none', use_softmax=False
            )
            loss = paddle.mean(loss)
            sgd = paddle.optimizer.SGD(learning_rate=0.001)
            sgd = paddle.static.amp.decorate(
                sgd, use_dynamic_loss_scaling=True, init_loss_scaling=128.0
            )
            sgd.minimize(loss)

        return loss

    def check(self, place, use_cuda):
        paddle.seed(1)
        paddle.framework.random._manual_program_seed(1)
        iters = 5
        batch_size = 16

        # build_fused_program: turn on fuse_bn_add_act_ops
        main_program = base.Program()
        startup_program = base.Program()
        loss = self.build_origin_program(
            main_program, startup_program, use_cuda
        )
        build_strategy_fused = base.BuildStrategy()
        build_strategy_fused.fuse_bn_add_act_ops = True
        binary_fused = base.CompiledProgram(
            main_program, build_strategy=build_strategy_fused
        )
        exe = base.Executor(place)
        loss_vals_fused = []
        x_data = []
        y_data = []
        scope = base.Scope()
        with base.scope_guard(scope):
            exe.run(startup_program)
            for _ in range(iters):
                x = np.random.random((batch_size, 1, 28, 28)).astype("float32")
                y = np.random.random((batch_size, 1)).astype("int64")
                x_data.append(x)
                y_data.append(y)
                loss_v = exe.run(
                    binary_fused, feed={"x": x, "y": y}, fetch_list=[loss]
                )
                loss_vals_fused.append(loss_v[0])

        # build_origin_program: turn off fused_bn_act_ops
        build_strategy = base.BuildStrategy()
        build_strategy.fuse_bn_add_act_ops = False
        binary = base.CompiledProgram(
            main_program, build_strategy=build_strategy_fused
        )
        loss_vals = []
        scope = base.Scope()
        with base.scope_guard(scope):
            exe.run(startup_program)
            for i in range(iters):
                loss_v = exe.run(
                    binary,
                    feed={"x": x_data[i], "y": y_data[i]},
                    fetch_list=[loss],
                )
                loss_vals.append(loss_v[0])

        # check loss
        for i in range(iters):
            self.assertAlmostEqual(loss_vals[i], loss_vals_fused[i], delta=1e-5)

    def test_fuse_bn_add_act(self):
        with paddle.pir_utils.OldIrGuard():
            place = base.CUDAPlace(0)
            self.check(place, use_cuda=True)

    def test_fuse_bn_add_act_API(self):
        with paddle.pir_utils.OldIrGuard():
            # build_fused_program: use fused_bn_add_act python API
            main_program = base.Program()
            startup_program = base.Program()
            place = base.CUDAPlace(0)
            x, y, loss = self.build_fused_program(
                main_program, startup_program, use_cuda=True
            )
            exe = base.Executor(place)
            scope = base.Scope()
            with base.scope_guard(scope):
                exe.run(startup_program)
                for _ in range(5):
                    x = np.random.random((4, 1, 28, 28)).astype("float32")
                    y = np.random.random((4, 1)).astype("int64")
                    loss_v = exe.run(
                        main_program, feed={"x": x, "y": y}, fetch_list=[loss]
                    )


if __name__ == '__main__':
    unittest.main()
