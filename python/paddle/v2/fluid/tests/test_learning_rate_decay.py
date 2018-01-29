# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

import math
import paddle.v2.fluid as fluid
import paddle.v2.fluid.layers as layers


class TestExponentialDecay(unittest.TestCase):
    def check_exponential_decay(self, staircase):
        init_lr = 1.0
        decay_steps = 5
        decay_rate = 0.5

        def exponential_decay(learning_rate,
                              global_step,
                              decay_steps,
                              decay_rate,
                              staircase=False):
            exponent = float(global_step) / float(decay_steps)
            if staircase:
                exponent = math.floor(exponent)
            return learning_rate * decay_rate**exponent

        global_step = layers.create_global_var(
            shape=[1], value=0.0, dtype='float32')

        decaied_lr = fluid.learning_rate_decay.exponential_decay(
            learning_rate=init_lr,
            global_step=global_step,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=staircase)
        layers.increment(global_step, 1.0)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        exe.run(fluid.default_startup_program())
        for step in range(10):
            step_val, lr_val = exe.run(fluid.default_main_program(),
                                       feed=[],
                                       fetch_list=[global_step, decaied_lr])
            python_decaied_lr = exponential_decay(
                learning_rate=init_lr,
                global_step=step,
                decay_steps=decay_steps,
                decay_rate=decay_rate,
                staircase=staircase)
            self.assertAlmostEqual(python_decaied_lr, lr_val[0])

    def test_exponential_decay_staircase(self):
        self.check_exponential_decay(True)

    def test_exponential_decay_staircase(self):
        self.check_exponential_decay(False)


if __name__ == '__main__':
    unittest.main()
