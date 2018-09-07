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

import unittest
import paddle.fluid.layers as layers
import paddle.fluid.core as core
from paddle.fluid.framework import default_startup_program, default_main_program
from paddle.fluid.executor import Executor
from paddle.fluid.backward import append_backward
import numpy


class ConditionalBlock(unittest.TestCase):
    def test_forward(self):
        data = layers.data(name='X', shape=[1], dtype='float32')
        data.stop_gradient = False
        cond = layers.ConditionalBlock(inputs=[data])
        out = layers.create_tensor(dtype='float32')
        with cond.block():
            hidden = layers.fc(input=data, size=10)
            layers.assign(hidden, out)

        cpu = core.CPUPlace()
        exe = Executor(cpu)
        exe.run(default_startup_program())

        x = numpy.random.random(size=(10, 1)).astype('float32')

        outs = exe.run(feed={'X': x}, fetch_list=[out])[0]
        print outs
        loss = layers.mean(out)
        append_backward(loss=loss)
        outs = exe.run(
            feed={'X': x},
            fetch_list=[
                default_main_program().block(0).var(data.name + "@GRAD")
            ])[0]
        print outs


if __name__ == '__main__':
    unittest.main()
