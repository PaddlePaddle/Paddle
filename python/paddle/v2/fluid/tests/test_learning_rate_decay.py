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

import paddle.v2.fluid as fluid
import paddle.v2.fluid.layers as layers


class TestExponentialDecay(unittest.TestCase):
    def test_exponential_decay(self):
        init_lr = 1.0
        decay_steps = 5
        decay_rate = 0.5

        def exponential_decay(step, lr):
            return lr * decay_rate**(step / decay_steps)

        global_step = layers.create_global_var(
            shape=[1], value=0.0, dtype='float32')
        global_lr = layers.create_global_var(
            shape=[1], value=init_lr, dtype='float32')
        layers.increment(global_step, 1.0)

        fluid.learning_rate_decay.exponential_decay(
            learning_rate=global_lr,
            global_step=global_step,
            decay_steps=decay_steps,
            decay_rate=decay_rate)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        exe.run(fluid.default_startup_program())
        for i in range(10):
            step_val, lr_val = exe.run(fluid.default_main_program(),
                                       feed=[],
                                       fetch_list=[global_step, global_lr])
            print(str(step_val) + ":" + str(lr_val))


if __name__ == '__main__':
    unittest.main()
