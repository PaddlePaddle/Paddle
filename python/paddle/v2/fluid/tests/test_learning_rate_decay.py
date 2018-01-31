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
import paddle.v2.fluid.framework as framework
import paddle.v2.fluid as fluid
import paddle.v2.fluid.layers as layers
import paddle.v2.fluid.learning_rate_decay as lr_decay


def exponential_decay(learning_rate,
                      global_step,
                      decay_steps,
                      decay_rate,
                      staircase=False):
    exponent = float(global_step) / float(decay_steps)
    if staircase:
        exponent = math.floor(exponent)
    return learning_rate * decay_rate**exponent


def natural_exp_decay(learning_rate,
                      global_step,
                      decay_steps,
                      decay_rate,
                      staircase=False):
    exponent = float(global_step) / float(decay_steps)
    if staircase:
        exponent = math.floor(exponent)
    return learning_rate * math.exp(-1 * decay_rate * exponent)


def inverse_time_decay(learning_rate,
                       global_step,
                       decay_steps,
                       decay_rate,
                       staircase=False):
    temp = float(global_step) / float(decay_steps)
    if staircase:
        temp = math.floor(temp)
    return learning_rate / (1 + decay_rate * temp)


class TestLearningRateDecay(unittest.TestCase):
    def check_decay(self, python_decay_fn, fluid_decay_fn, staircase):
        init_lr = 1.0
        decay_steps = 5
        decay_rate = 0.5

        global_step = layers.create_global_var(
            shape=[1], value=0.0, dtype='float32', persistable=True)

        decayed_lr = fluid_decay_fn(
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
                                       fetch_list=[global_step, decayed_lr])
            python_decayed_lr = python_decay_fn(
                learning_rate=init_lr,
                global_step=step,
                decay_steps=decay_steps,
                decay_rate=decay_rate,
                staircase=staircase)
            self.assertAlmostEqual(python_decayed_lr, lr_val[0])

    def test_decay(self):
        decay_fns = [
            (exponential_decay, lr_decay.exponential_decay, True),
            (exponential_decay, lr_decay.exponential_decay, False),
            (natural_exp_decay, lr_decay.natural_exp_decay, True),
            (natural_exp_decay, lr_decay.natural_exp_decay, False),
            (inverse_time_decay, lr_decay.inverse_time_decay, True),
            (inverse_time_decay, lr_decay.inverse_time_decay, False),
        ]

        for py_decay_fn, fluid_decay_fn, staircase in decay_fns:
            print("decay_fn=" + str(py_decay_fn) + " staircase=" + str(
                staircase))
            main_program = framework.Program()
            startup_program = framework.Program()
            with framework.program_guard(main_program, startup_program):
                self.check_decay(py_decay_fn, fluid_decay_fn, staircase)


if __name__ == '__main__':
    unittest.main()
