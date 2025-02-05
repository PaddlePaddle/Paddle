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

import numpy

import paddle
import paddle.nn.functional as F
from paddle import base


class TestFuseActElewiseAddInplaceGradPass(unittest.TestCase):
    def build_program(self, main_program, startup_program):
        with paddle.static.program_guard(main_program, startup_program):
            X = paddle.static.data(name="X", shape=[3, 3], dtype='float32')
            Y = paddle.static.data(name="Y", shape=[3, 3], dtype='float32')
            Out1 = X * 5
            Out2 = F.relu(Out1)
            prediction = paddle.tensor.math._add_with_axis(Y, Out2, axis=1)
            loss = paddle.mean(prediction)
            sgd = paddle.optimizer.SGD(learning_rate=0.001)
            sgd.minimize(loss)
        return X, Y, loss

    def check(self, place):
        paddle.seed(1)
        numpy.random.seed(1)
        paddle.framework.random._manual_program_seed(1)
        main_program = base.Program()
        startup_program = base.Program()
        X, Y, loss = self.build_program(main_program, startup_program)
        exe = base.Executor(place)

        x = numpy.random.random(size=(3, 3)).astype('float32')
        y = numpy.random.random(size=(3, 3)).astype('float32')
        label = numpy.random.random(size=(3, 3)).astype('float32')

        # open fused_pass
        build_strategy = base.BuildStrategy()
        build_strategy.fuse_elewise_add_act_ops = True
        compiled_prog_fused = paddle.static.CompiledProgram(
            main_program, build_strategy=build_strategy
        )
        scope = base.Scope()
        with base.scope_guard(scope):
            exe.run(startup_program)
            loss_data_fused = exe.run(
                compiled_prog_fused,
                feed={"X": x, "Y": y},
                fetch_list=[loss],
            )

        # close fused_pass
        build_strategy = base.BuildStrategy()
        build_strategy.fuse_elewise_add_act_ops = False
        compiled_prog = paddle.static.CompiledProgram(
            main_program, build_strategy=build_strategy
        )
        scope = base.Scope()
        with base.scope_guard(scope):
            exe.run(startup_program)
            loss_data = exe.run(
                compiled_prog, feed={"X": x, "Y": y}, fetch_list=[loss]
            )

        self.assertEqual(loss_data_fused, loss_data)

    def test_fuse_act_add_grad_pass_cpu(self):
        paddle.enable_static()
        place = base.CPUPlace()
        self.check(place)

    def test_fuse_act_add_grad_pass_cuda(self):
        if base.core.is_compiled_with_cuda():
            place = base.CUDAPlace(0)
            self.check(place)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
