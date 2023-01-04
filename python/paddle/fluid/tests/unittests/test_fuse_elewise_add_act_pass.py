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

import os
import unittest

import numpy
from parallel_executor_test_base import DeviceType, TestParallelExecutorBase
from simple_nets import fc_with_batchnorm, init_data, simple_fc_net

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.nn.functional as F


class TestMNIST(TestParallelExecutorBase):
    @classmethod
    def setUpClass(cls):
        os.environ['CPU_NUM'] = str(4)

    def _compare_fuse_elewise_add_act_ops(self, model, use_device):
        if use_device == DeviceType.CUDA and not core.is_compiled_with_cuda():
            return
        img, label = init_data()

        def _optimizer(learning_rate=1e-6):
            optimizer = fluid.optimizer.SGD(
                learning_rate=learning_rate,
                regularization=fluid.regularizer.L2Decay(1e-6),
            )
            return optimizer

        # NOTE(dzh):
        # need to make it compatible with elewise fuse act
        # FIXME (liuwei12)
        # the new memory optimize strategy will crash this unittest
        # add enable_inplace=False here to force pass the unittest
        (
            not_fuse_op_first_loss,
            not_fuse_op_last_loss,
            _,
        ) = self.check_network_convergence(
            model,
            feed_dict={"image": img, "label": label},
            use_device=use_device,
            fuse_elewise_add_act_ops=False,
            use_ir_memory_optimize=False,
            enable_inplace=False,
            optimizer=_optimizer,
        )
        (
            fuse_op_first_loss,
            fuse_op_last_loss,
            _,
        ) = self.check_network_convergence(
            model,
            feed_dict={"image": img, "label": label},
            use_device=use_device,
            fuse_elewise_add_act_ops=True,
            use_ir_memory_optimize=False,
            enable_inplace=False,
            optimizer=_optimizer,
        )

        for loss in zip(not_fuse_op_first_loss, fuse_op_first_loss):
            self.assertAlmostEquals(loss[0], loss[1], delta=1e-6)
        for loss in zip(not_fuse_op_last_loss, fuse_op_last_loss):
            self.assertAlmostEquals(loss[0], loss[1], delta=1e-6)

    def test_simple_fc_with_fuse_op(self):
        self._compare_fuse_elewise_add_act_ops(simple_fc_net, DeviceType.CUDA)
        self._compare_fuse_elewise_add_act_ops(simple_fc_net, DeviceType.CPU)

    def test_batchnorm_fc_with_fuse_op(self):
        self._compare_fuse_elewise_add_act_ops(
            fc_with_batchnorm, DeviceType.CUDA
        )
        self._compare_fuse_elewise_add_act_ops(
            fc_with_batchnorm, DeviceType.CPU
        )


class TestFuseActElewiseAddInplaceGradPass(unittest.TestCase):
    def build_program(self, main_program, startup_program):
        with paddle.static.program_guard(main_program, startup_program):
            X = fluid.data(name="X", shape=[3, 3], dtype='float32')
            Y = fluid.data(name="Y", shape=[3, 3], dtype='float32')
            Out1 = X * 5
            Out2 = F.relu(Out1)
            prediction = paddle.tensor.math._add_with_axis(Y, Out2, axis=1)
            loss = paddle.mean(prediction)
            sgd = fluid.optimizer.SGD(learning_rate=0.001)
            sgd.minimize(loss)
        return X, Y, loss

    def check(self, place):
        paddle.seed(1)
        numpy.random.seed(1)
        paddle.framework.random._manual_program_seed(1)
        main_program = fluid.Program()
        startup_program = fluid.Program()
        X, Y, loss = self.build_program(main_program, startup_program)
        exe = fluid.Executor(place)

        x = numpy.random.random(size=(3, 3)).astype('float32')
        y = numpy.random.random(size=(3, 3)).astype('float32')
        label = numpy.random.random(size=(3, 3)).astype('float32')

        # open fused_pass
        build_strategy = fluid.BuildStrategy()
        build_strategy.fuse_elewise_add_act_ops = True
        compiled_prog_fused = paddle.static.CompiledProgram(
            main_program, build_strategy=build_strategy
        )
        scope = fluid.Scope()
        with fluid.scope_guard(scope):
            exe.run(startup_program)
            loss_data_fused = exe.run(
                compiled_prog_fused,
                feed={"X": x, "Y": y},
                fetch_list=[loss.name],
            )

        # close fused_pass
        build_strategy = fluid.BuildStrategy()
        build_strategy.fuse_elewise_add_act_ops = False
        compiled_prog = paddle.static.CompiledProgram(
            main_program, build_strategy=build_strategy
        )
        scope = fluid.Scope()
        with fluid.scope_guard(scope):
            exe.run(startup_program)
            loss_data = exe.run(
                compiled_prog, feed={"X": x, "Y": y}, fetch_list=[loss.name]
            )

        self.assertEqual(loss_data_fused, loss_data)

    def test_fuse_act_add_grad_pass_cpu(self):
        place = fluid.CPUPlace()
        self.check(place)

    def test_fuse_act_add_grad_pass_cuda(self):
        if fluid.core.is_compiled_with_cuda():
            place = fluid.CUDAPlace(0)
            self.check(place)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
