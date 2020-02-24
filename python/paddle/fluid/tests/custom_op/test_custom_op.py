# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import sys
import numpy as np
import unittest
import contextlib

import paddle
import paddle.fluid as fluid

file_dir = os.path.dirname(os.path.abspath(__file__))
fluid.load_op_library(os.path.join(file_dir, 'librelu2_op.so'))

from paddle.fluid.layer_helper import LayerHelper


def relu2(x, name=None):
    helper = LayerHelper("relu2", **locals())
    out = helper.create_variable(
        type=x.type, name=name, dtype=x.dtype, persistable=False)
    helper.append_op(type="relu2", inputs={"X": x}, outputs={"Y": out})
    return out


@contextlib.contextmanager
def scope_prog_guard():
    prog = fluid.Program()
    startup_prog = fluid.Program()
    scope = fluid.core.Scope()
    with fluid.scope_guard(scope):
        with fluid.program_guard(prog, startup_prog):
            yield


def linear_fc(data, label, use_custom_relu):
    hidden = fluid.layers.fc(data, size=128)
    hidden = relu2(hidden) if use_custom_relu else fluid.layers.relu(hidden)
    hidden = fluid.layers.fc(hidden, size=128)
    hidden = fluid.layers.fc(hidden, size=10, act='softmax')
    loss = fluid.layers.cross_entropy(input=hidden, label=label)
    loss = fluid.layers.mean(loss)
    return loss


def custom_op_test(use_gpu=True, use_custom_relu=True):
    with scope_prog_guard():
        np.random.seed(0)
        fluid.default_startup_program().random_seed = 10
        fluid.default_main_program().random_seed = 10

        data = fluid.layers.data(
            name='data', shape=[1, 28, 28], dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        loss = linear_fc(data, label, use_custom_relu)

        optimizer = fluid.optimizer.Momentum(learning_rate=0.1, momentum=0.9)
        optimizer.minimize(loss)

        place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())

        compile_program = fluid.compiler.CompiledProgram(
            fluid.default_main_program()).with_data_parallel(
                loss_name=loss.name)

        reader = paddle.batch(paddle.dataset.mnist.train(), batch_size=32)
        feeder = fluid.DataFeeder(feed_list=[data, label], place=place)

        num = 4
        for i, data in enumerate(reader()):
            outs, = exe.run(compile_program,
                            feed=feeder.feed(data),
                            fetch_list=[loss])
            if i == num:
                break
        return outs


class CustomOpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ['CPU_NUM'] = str(2)

    def test_cpu(self):
        actual = custom_op_test(False, True)
        expect = custom_op_test(False, False)
        self.assertEqual(actual.all(), expect.all())

    def test_gpu(self):
        if not fluid.core.is_compiled_with_cuda():
            return
        actual = custom_op_test(True, True)
        expect = custom_op_test(True, False)
        self.assertEqual(actual.all(), expect.all())


if __name__ == '__main__':
    unittest.main()
