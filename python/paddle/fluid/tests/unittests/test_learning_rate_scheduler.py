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

        for step in range(10):
            lr_val, = exe.run(main_prog, feed={}, fetch_list=[decayed_lr])
            python_decayed_lr = python_decay_fn(
                global_step=float(step), **kwargs)
            self.assertAlmostEqual(
                python_decayed_lr,
                lr_val[0],
                msg='Failed lr scheduler is {0}, step {1}, Python result is {2}, Fluid result is {3}'.
                format(python_decay_fn.__name__,
                       str(step), str(python_decayed_lr), str(lr_val[0])))

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
            print("class=" + self.__class__.__name__ + "decay_fn=" +
                  py_decay_fn.__name__ + " kwargs=" + str(kwargs))
            main_program = framework.Program()
            startup_program = framework.Program()
            with framework.program_guard(main_program, startup_program):
                self.check_decay(py_decay_fn, fluid_decay_fn, kwargs)


def linear_lr_warmup(global_step, warmup_steps, start_lr, end_lr):
    linear_step = end_lr - start_lr
    decayed_lr = start_lr + linear_step * (global_step / warmup_steps)
    return decayed_lr


class TestLinearWamrupLearningRateDecay(TestLearningRateDecay):
    def check_decay_with_place(self, place, python_decay_fn, fluid_decay_fn,
                               kwargs):
        main_prog = fluid.Program()
        startup_prog = fluid.Program()

        warmup_steps = 10
        start_lr = 0.1 / 3.
        end_lr = 0.1

        with fluid.program_guard(main_prog, startup_prog):
            decayed_lr = layers.linear_lr_warmup(
                fluid_decay_fn(**kwargs), warmup_steps, start_lr, end_lr)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_prog)

        for step in range(20):
            lr_val, = exe.run(main_prog, feed={}, fetch_list=[decayed_lr])
            if step < warmup_steps:
                python_decayed_lr = linear_lr_warmup(
                    float(step), warmup_steps, start_lr, end_lr)
            else:
                python_decayed_lr = python_decay_fn(
                    global_step=float(step), **kwargs)
            self.assertAlmostEqual(
                python_decayed_lr,
                lr_val[0],
                msg='Test {0} Failed, step {1}, Python result is {2}, Fluid result is {3}'.
                format(python_decay_fn.__name__,
                       str(step), str(python_decayed_lr), str(lr_val[0])))


class TestLinearWamrupLearningRateDecayWithScalarInput(unittest.TestCase):
    def run_scalar_lr(self, place, lr, start_lr, end_lr):
        main_prog = fluid.Program()
        startup_prog = fluid.Program()

        warmup_steps = 10

        with fluid.program_guard(main_prog, startup_prog):
            decayed_lr = layers.linear_lr_warmup(lr, warmup_steps, start_lr,
                                                 end_lr)

        exe = fluid.Executor(place)
        exe.run(startup_prog)

        for step in range(20):
            lr_val, = exe.run(main_prog, feed={}, fetch_list=[decayed_lr])
            if step < warmup_steps:
                expected_lr = linear_lr_warmup(
                    float(step), warmup_steps, start_lr, end_lr)
            else:
                expected_lr = lr
            self.assertAlmostEqual(
                expected_lr,
                lr_val[0],
                msg='Test failed, step {0}, expected {1}, but got {2}'.format(
                    step, expected_lr, lr_val[0]))

    def test_scalar_lr(self):
        def run_places(lr, start_lr, end_lr):
            places = [fluid.CPUPlace()]
            if core.is_compiled_with_cuda():
                places.append(fluid.CUDAPlace(0))
            for p in places:
                self.run_scalar_lr(p, lr, start_lr, end_lr)

        # float
        lr = 0.2
        start_lr = 0.1 / 3.
        end_lr = 0.2
        run_places(lr, start_lr, end_lr)

        # int end_lr
        lr = 2.
        start_lr = 0.1 / 3.
        end_lr = 1
        run_places(lr, start_lr, end_lr)

        # int
        lr = 1
        start_lr = 0
        end_lr = 1
        run_places(lr, start_lr, end_lr)


class TestLinearWamrupLearningRateDecayDygraphMode(unittest.TestCase):
    def test_dygraph_mode(self):
        with fluid.dygraph.guard():
            lr = fluid.layers.polynomial_decay(
                learning_rate=1.0,
                decay_steps=10,
                end_learning_rate=0.0,
                power=1.0)
            lr = fluid.layers.linear_lr_warmup(
                learning_rate=lr, warmup_steps=2, start_lr=0.0, end_lr=1.0)

            right_result = [0.5, 0.9, 0.8, 0.7, 0.6]
            for i in range(5):

                t = lr()

                self.assertEqual(t[0], right_result[i])


class TestLinearWamrupLearningRateDecayDygraphModeTypeCheck(unittest.TestCase):
    def test_dygraph_mode(self):
        with fluid.dygraph.guard():
            with self.assertRaises(TypeError):
                lr = fluid.layers.linear_lr_warmup(
                    learning_rate="fake_lr",
                    warmup_steps=2,
                    start_lr=0.0,
                    end_lr=1.0)


if __name__ == '__main__':
    unittest.main()
