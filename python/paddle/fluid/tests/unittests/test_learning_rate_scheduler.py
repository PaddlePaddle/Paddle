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

from __future__ import print_function

import copy
import math
import unittest

import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.framework as framework
import paddle.fluid.core as core


def exponential_decay(learning_rate,
                      global_step,
                      decay_steps,
                      decay_rate,
                      staircase=False):
    exponent = global_step / decay_steps
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


def polynomial_decay(learning_rate,
                     global_step,
                     decay_steps,
                     end_learning_rate=0.0001,
                     power=1.0,
                     cycle=False):
    if cycle:
        div = math.ceil(global_step / float(decay_steps))
        if div == 0:
            div = 1
        decay_steps = decay_steps * div
    else:
        global_step = min(global_step, decay_steps)
    return (learning_rate - end_learning_rate) * \
           ((1 - float(global_step) / float(decay_steps)) ** power) + end_learning_rate


def piecewise_decay(global_step, boundaries, values):
    assert len(boundaries) + 1 == len(values)
    for i in range(len(boundaries)):
        if global_step < boundaries[i]:
            return values[i]
    return values[len(values) - 1]


def cosine_decay(global_step, learning_rate, step_each_epoch, epochs):
    cur_epoch = math.floor(global_step / step_each_epoch)
    decayed_lr = learning_rate * 0.5 * (
        math.cos(cur_epoch * math.pi / epochs) + 1)
    return decayed_lr


class TestLearningRateDecay(unittest.TestCase):
    def check_decay(self, python_decay_fn, fluid_decay_fn, kwargs):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for place in places:
            self.check_decay_with_place(place, python_decay_fn, fluid_decay_fn,
                                        kwargs)

    def check_decay_with_place(self, place, python_decay_fn, fluid_decay_fn,
                               kwargs):
        main_prog = fluid.Program()
        startup_prog = fluid.Program()

        with fluid.program_guard(main_prog, startup_prog):
            decayed_lr = fluid_decay_fn(**kwargs)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        exe.run(startup_prog)

        fluid.memory_optimize(main_prog)

        for step in range(10):
            lr_val, = exe.run(main_prog, feed={}, fetch_list=[decayed_lr])
            python_decayed_lr = python_decay_fn(
                global_step=float(step), **kwargs)
            self.assertAlmostEqual(
                python_decayed_lr,
                lr_val[0],
                msg='Failed fn is {0}, Python result is {1}, Fluid result is {2}'.
                format(python_decay_fn.__name__,
                       str(python_decayed_lr), str(lr_val[0])))

    def test_decay(self):
        common_kwargs_true = {
            "learning_rate": 1.0,
            "decay_steps": 5,
            "decay_rate": 0.5,
            "staircase": True
        }
        common_kwargs_false = copy.deepcopy(common_kwargs_true)
        common_kwargs_false["staircase"] = False

        decay_fns = [
            (exponential_decay, layers.exponential_decay, common_kwargs_true),
            (exponential_decay, layers.exponential_decay, common_kwargs_false),
            (natural_exp_decay, layers.natural_exp_decay, common_kwargs_true),
            (natural_exp_decay, layers.natural_exp_decay, common_kwargs_false),
            (inverse_time_decay, layers.inverse_time_decay, common_kwargs_true),
            (inverse_time_decay, layers.inverse_time_decay,
             common_kwargs_false),
            (polynomial_decay, layers.polynomial_decay, {
                "learning_rate": 1.0,
                "decay_steps": 5,
                "cycle": True
            }),
            (polynomial_decay, layers.polynomial_decay, {
                "learning_rate": 1.0,
                "decay_steps": 5,
                "cycle": False
            }),
            (piecewise_decay, layers.piecewise_decay, {
                "boundaries": [3, 6, 9],
                "values": [0.1, 0.2, 0.3, 0.4]
            }),
            (cosine_decay, layers.cosine_decay, {
                "learning_rate": 0.1,
                "step_each_epoch": 100,
                "epochs": 120
            }),
        ]

        for py_decay_fn, fluid_decay_fn, kwargs in decay_fns:
            print("decay_fn=" + py_decay_fn.__name__ + " kwargs=" + str(kwargs))
            main_program = framework.Program()
            startup_program = framework.Program()
            with framework.program_guard(main_program, startup_program):
                self.check_decay(py_decay_fn, fluid_decay_fn, kwargs)


if __name__ == '__main__':
    unittest.main()
