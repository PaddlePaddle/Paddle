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
import numpy as np
import unittest

import paddle
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


def noam_decay(global_step, d_model, warmup_steps, learning_rate=1.0):
    a = math.pow(global_step, -0.5)
    b = math.pow(warmup_steps, -1.5) * global_step
    decayed_lr = learning_rate * math.pow(d_model, -0.5) * min(a, b)

    return decayed_lr


def linear_lr_warmup(global_step, warmup_steps, start_lr, end_lr):
    linear_step = end_lr - start_lr
    decayed_lr = start_lr + linear_step * (global_step / warmup_steps)
    return decayed_lr


def multi_step_decay(global_step, learning_rate, milestones, decay_rate=0.1):
    for i in range(len(milestones)):
        if global_step < milestones[i]:
            return learning_rate * math.pow(decay_rate, i)

    return learning_rate * math.pow(decay_rate, len(milestones))


def step_decay(global_step, learning_rate, step_size, decay_rate=0.1):
    return learning_rate * math.pow(decay_rate, global_step // step_size)


def lambda_decay(global_step, learning_rate, lr_lambda):
    return learning_rate * lr_lambda(global_step)


class TestLearningRateDecayDygraph(unittest.TestCase):
    def test_LR_state_dict(self):
        with fluid.dygraph.guard():
            x = np.random.uniform(-1, 1, [3, 10]).astype("float32")
            linear = fluid.dygraph.Linear(10, 10)
            input = fluid.dygraph.to_variable(x)

            Exponential_scheduler = fluid.dygraph.ExponentialDecay(
                learning_rate=0.1,
                decay_steps=10000,
                decay_rate=0.5,
                staircase=True)
            Step_scheduler = fluid.dygraph.StepDecay(0.5, step_size=3)
            Reducelr_scheduler = fluid.dygraph.ReduceLROnPlateau(
                learning_rate=1.0, decay_rate=0.5, patience=5, cooldown=3)

            adam1 = fluid.optimizer.Adam(
                learning_rate=Exponential_scheduler,
                parameter_list=linear.parameters())
            adam2 = fluid.optimizer.Adam(
                learning_rate=Step_scheduler,
                parameter_list=linear.parameters())
            adam3 = fluid.optimizer.Adam(
                learning_rate=Reducelr_scheduler,
                parameter_list=linear.parameters())
            print(adam3.state_dict())

            for epoch in range(10):
                out = linear(input)
                loss = fluid.layers.reduce_mean(out)
                loss.backward()
                adam1.minimize(loss)
                adam2.minimize(loss)
                adam3.minimize(loss)
                linear.clear_gradients()

                Step_scheduler.epoch()
                Reducelr_scheduler.step(loss)

            fluid.dygraph.save_dygraph(linear.state_dict(), "save_path")

            Exponential_scheduler_test = fluid.dygraph.ExponentialDecay(
                learning_rate=0.1,
                decay_steps=10000,
                decay_rate=0.5,
                staircase=True)
            Step_scheduler_test = fluid.dygraph.StepDecay(0.5, step_size=3)
            Reducelr_scheduler_test = fluid.dygraph.ReduceLROnPlateau(
                learning_rate=1.0, decay_rate=0.5, patience=5, cooldown=3)

            fluid.dygraph.save_dygraph(adam1.state_dict(), "save_path")
            _, opt_state = fluid.dygraph.load_dygraph("save_path")
            adam_test = fluid.optimizer.Adam(
                learning_rate=Exponential_scheduler_test,
                parameter_list=linear.parameters())
            adam_test.set_dict(opt_state)
            self.assertEqual(adam_test._learning_rate.step_num,
                             adam1._learning_rate.step_num,
                             "epoch_num is different before and after set_dict")

            fluid.dygraph.save_dygraph(adam2.state_dict(), "save_path")
            _, opt_state = fluid.dygraph.load_dygraph("save_path")
            adam_test = fluid.optimizer.Adam(
                learning_rate=Step_scheduler_test,
                parameter_list=linear.parameters())
            adam_test.set_dict(opt_state)
            self.assertEqual(adam_test._learning_rate.epoch_num,
                             adam2._learning_rate.epoch_num,
                             "epoch_num is different before and after set_dict")
            self.assertEqual(
                adam_test._learning_rate(),
                adam2._learning_rate(),
                "current learning rate is different before and after set_dict")

            fluid.dygraph.save_dygraph(adam3.state_dict(), "save_path")
            _, opt_state = fluid.dygraph.load_dygraph("save_path")
            adam_test = fluid.optimizer.Adam(
                learning_rate=Reducelr_scheduler_test,
                parameter_list=linear.parameters())
            adam_test.set_dict(opt_state)
            self.assertEqual(adam_test._learning_rate.best_loss,
                             adam3._learning_rate.best_loss.numpy()[0],
                             "best_loss is different before and after set_dict")
            self.assertEqual(
                adam_test._learning_rate.cooldown_counter,
                adam3._learning_rate.cooldown_counter,
                "cooldown_counter is different before and after set_dict")
            self.assertEqual(
                adam_test._learning_rate.num_bad_epochs,
                adam3._learning_rate.num_bad_epochs,
                "num_bad_epochs is different before and after set_dict")
            self.assertEqual(adam_test._learning_rate.epoch_num,
                             adam3._learning_rate.epoch_num,
                             "epoch is different before and after set_dict")
            self.assertEqual(
                adam_test._learning_rate(),
                adam3._learning_rate(),
                "current learning rate is different before and after set_dict")

    def test_NoamDecay(self):
        with fluid.dygraph.guard():
            d_model = 0.01
            warmup_steps = 200
            learning_rate = 2.0
            lr = fluid.layers.noam_decay(d_model, warmup_steps, learning_rate)
            for step in range(5):
                step += 1
                right_result = noam_decay(step, d_model, warmup_steps,
                                          learning_rate)
                fluid_result = lr()

                self.assertAlmostEqual(
                    right_result,
                    fluid_result[0],
                    msg='Failed lr scheduler in step {0}, Python result is {1}, Fluid result is {2}'.
                    format(step, right_result, fluid_result[0]))

    def test_LinearLrWarmup(self):
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

                self.assertTrue(
                    np.allclose((t.numpy())[0].item(), right_result[i]))

            with self.assertRaises(TypeError):
                lr = fluid.layers.linear_lr_warmup(
                    learning_rate="fake_lr",
                    warmup_steps=2,
                    start_lr=0.0,
                    end_lr=1.0)

    def test_MultiStepDecay(self):
        with fluid.dygraph.guard():
            learning_rate = 0.5
            milestones = [2, 4, 8]
            decay_rate = 0.2
            linear = fluid.dygraph.Linear(10, 10)

            scheduler = fluid.dygraph.MultiStepDecay(learning_rate, milestones,
                                                     decay_rate)

            adam = fluid.optimizer.AdamOptimizer(
                learning_rate=scheduler, parameter_list=linear.parameters())
            for epoch in range(10):
                right_result = multi_step_decay(epoch, learning_rate,
                                                milestones, decay_rate)
                fluid_result = adam.current_step_lr()
                scheduler.epoch()
                self.assertAlmostEqual(
                    right_result,
                    fluid_result,
                    msg='Failed lr scheduler in epoch {0}, Python result is {1}, Fluid result is {2}'.
                    format(epoch, right_result, fluid_result))

            with self.assertRaises(ValueError):
                lr = fluid.dygraph.MultiStepDecay(learning_rate, [30, 50, 20],
                                                  0.1)

            with self.assertRaises(ValueError):
                lr = fluid.dygraph.MultiStepDecay(learning_rate, [20, 30, 50],
                                                  1)

            with self.assertRaises(TypeError):
                lr = fluid.dygraph.MultiStepDecay("test", [20, 30, 50])

            with self.assertRaises(ValueError):
                lr = fluid.dygraph.MultiStepDecay(-1, [20, 30, 50])

    def test_StepDecay(self):
        with fluid.dygraph.guard():
            learning_rate = 0.5
            step_size = 3
            decay_rate = 0.2
            scheduler = fluid.dygraph.StepDecay(learning_rate, step_size,
                                                decay_rate)
            for epoch in range(10):
                right_result = step_decay(epoch, learning_rate, step_size,
                                          decay_rate)
                fluid_result = scheduler().numpy()[0]
                scheduler.epoch()
                self.assertAlmostEqual(
                    right_result,
                    fluid_result,
                    msg='Failed lr scheduler in epoch {0}, Python result is {1}, Fluid result is {2}'.
                    format(epoch, right_result, fluid_result))

            with self.assertRaises(TypeError):
                lr = fluid.dygraph.StepDecay(learning_rate, "test", 0.1)

            with self.assertRaises(ValueError):
                lr = fluid.dygraph.StepDecay(learning_rate, 20, 2)

    def test_LambdaDecay(self):
        with fluid.dygraph.guard():
            learning_rate = 0.5
            lr_lambda = lambda x: 0.95**x
            scheduler = fluid.dygraph.LambdaDecay(learning_rate, lr_lambda)

            linear = fluid.dygraph.nn.Linear(10, 10)
            adam = fluid.optimizer.Adam(
                scheduler, parameter_list=linear.parameters())

            for epoch in range(30):
                right_result = lambda_decay(epoch, learning_rate, lr_lambda)
                fluid_result = scheduler().numpy()[0]
                scheduler.epoch()
                self.assertAlmostEqual(
                    right_result,
                    fluid_result,
                    msg='Failed lr scheduler in epoch {0}, Python result is {1}, Fluid result is {2}'.
                    format(epoch, right_result, fluid_result))

            with self.assertRaises(TypeError):
                lr = fluid.dygraph.LambdaDecay(learning_rate, "test")


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
            # Step of NoamDecay starts from 1.
            if python_decay_fn.__name__ == 'noam_decay':
                step += 1
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
             common_kwargs_false), (polynomial_decay, layers.polynomial_decay, {
                 "learning_rate": 1.0,
                 "decay_steps": 5,
                 "cycle": True
             }), (polynomial_decay, layers.polynomial_decay, {
                 "learning_rate": 1.0,
                 "decay_steps": 5,
                 "cycle": False
             }), (piecewise_decay, layers.piecewise_decay, {
                 "boundaries": [3, 6, 9],
                 "values": [0.1, 0.2, 0.3, 0.4]
             }), (cosine_decay, layers.cosine_decay, {
                 "learning_rate": 0.1,
                 "step_each_epoch": 100,
                 "epochs": 120
             }), (noam_decay, layers.noam_decay, {
                 "d_model": 0.01,
                 "warmup_steps": 200,
                 "learning_rate": 2.0
             })
        ]

        for py_decay_fn, fluid_decay_fn, kwargs in decay_fns:
            print("class=" + self.__class__.__name__ + " decay_fn=" +
                  py_decay_fn.__name__ + " kwargs=" + str(kwargs))
            main_program = framework.Program()
            startup_program = framework.Program()
            with framework.program_guard(main_program, startup_program):
                self.check_decay(py_decay_fn, fluid_decay_fn, kwargs)


class TestLinearWamrupLearningRateDecay(unittest.TestCase):
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
            # Step of NoamDecay starts from 1.
            if fluid_decay_fn.__name__ == 'noam_decay':
                step += 1
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


def reduce_lr_on_plateau(decay_rate, threshold, cooldown, patience, m, n, loss,
                         var_list):
    def is_better(current, best, m, n):
        if m == 'min' and n == 'rel':
            return current < best - best * threshold
        elif m == 'min' and n == 'abs':
            return current < best - threshold
        elif m == 'max' and n == 'rel':
            return current > best + best * threshold
        else:  # mode == 'max' and epsilon_mode == 'abs':
            return current > best + threshold

    if var_list[2] > 0:
        var_list[2] -= 1
        return var_list[1]

    if is_better(loss, var_list[0], m, n):
        var_list[0] = loss
        var_list[3] = 0
    else:
        var_list[3] += 1
        if var_list[3] > patience:
            var_list[2] = cooldown
            var_list[3] = 0
            new_lr = var_list[1] * decay_rate
            var_list[1] = new_lr if var_list[1] - new_lr > 1e-8 else var_list[1]

    return var_list[1]


class TestReduceLROnPlateauDecay(unittest.TestCase):
    def test_ReduceLR(self):
        # the decay rate must be less than 1.0
        with self.assertRaises(ValueError):
            paddle.optimizer.ReduceLROnPlateau(learning_rate=1.0, factor=2.0)
        # the mode must be "min" or "max"
        with self.assertRaises(ValueError):
            paddle.optimizer.ReduceLROnPlateau(learning_rate=1.0, mode="test")
        # the threshold_mode must be "rel" or "abs"
        with self.assertRaises(ValueError):
            paddle.optimizer.ReduceLROnPlateau(
                learning_rate=1.0, threshold_mode="test")
        with self.assertRaises(TypeError):
            paddle.optimizer.ReduceLROnPlateau(learning_rate="test")
        with self.assertRaises(TypeError):
            paddle.optimizer.ReduceLROnPlateau(learning_rate=0.5).step("test")

        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))

        for place in places:
            for m, n in zip(['min', 'max', 'min', 'max'],
                            ['rel', 'rel', 'abs', 'abs']):
                kwargs = {
                    'learning_rate': 1.0,
                    'mode': m,
                    'factor': 0.5,
                    'patience': 3,
                    'threshold': 1e-4,
                    'threshold_mode': n,
                    'cooldown': 1,
                    'min_lr': 0,
                    'epsilon': 1e-8,
                    'verbose': False,
                }
                paddle.enable_static()
                self._test_static(place, kwargs)
                paddle.disable_static(place)
                self._test_dygraph(place, kwargs)
                paddle.enable_static()

    def _test_static(self, place, kwargs):
        paddle.enable_static()

        best = float("-10000") if kwargs['mode'] == "max" else float("10000")
        current_lr = 1.0
        cooldown_counter = 0
        num_bad_epochs = 0
        var_list = [best, current_lr, cooldown_counter, num_bad_epochs]

        main_prog = fluid.Program()
        start_prog = fluid.Program()
        with fluid.program_guard(main_prog, start_prog):
            x = fluid.layers.create_global_var(
                [1], 1, 'float32', persistable=True)
            paddle.increment(x)
            loss = paddle.sin(x)
            scheduler = paddle.optimizer.ReduceLROnPlateau(**kwargs)
            adam = fluid.optimizer.Adam(learning_rate=scheduler)
            adam.minimize(loss)
            lr_var = adam._global_learning_rate()
            test_prog = main_prog.clone()

        exe = fluid.Executor(place)
        exe.run(start_prog)

        for epoch in range(20):
            for batch_id in range(1):
                out, actual_lr = exe.run(main_prog,
                                         fetch_list=[loss.name, lr_var.name])
                expected_lr = reduce_lr_on_plateau(
                    kwargs['factor'], kwargs['threshold'], kwargs['cooldown'],
                    kwargs['patience'], kwargs['mode'],
                    kwargs['threshold_mode'], out[0], var_list)

            scheduler.step(out[0])
            actual_lr = scheduler()
            self.assertEqual(actual_lr, np.array(expected_lr))

        for epoch in range(10):
            for batch_id in range(1):
                out, actual_lr = exe.run(test_prog,
                                         fetch_list=[loss.name, lr_var.name])
                expected_lr = reduce_lr_on_plateau(
                    kwargs['factor'], kwargs['threshold'], kwargs['cooldown'],
                    kwargs['patience'], kwargs['mode'],
                    kwargs['threshold_mode'], out[0], var_list)
            scheduler.step(out[0])
            actual_lr = scheduler()
            self.assertEqual(actual_lr, np.array(expected_lr))

    def _test_dygraph(self, place, kwargs):
        paddle.disable_static(place)

        best = float("-10000") if kwargs['mode'] == "max" else float("10000")
        current_lr = 1.0
        cooldown_counter = 0
        num_bad_epochs = 0
        var_list = [best, current_lr, cooldown_counter, num_bad_epochs]

        linear = paddle.nn.Linear(10, 10)
        scheduler = paddle.optimizer.ReduceLROnPlateau(**kwargs)
        sgd = paddle.optimizer.SGD(learning_rate=scheduler,
                                   parameter_list=linear.parameters())

        for epoch in range(20):
            for batch_id in range(1):
                x = paddle.to_tensor(epoch).astype('float32')
                loss = paddle.sin(x)
                loss.backward()
                sgd.minimize(loss)

            scheduler.step(loss)
            # get lr from paddle
            current_lr = scheduler()
            # get lr form python
            expected_lr = reduce_lr_on_plateau(
                kwargs['factor'], kwargs['threshold'], kwargs['cooldown'],
                kwargs['patience'], kwargs['mode'], kwargs['threshold_mode'],
                loss, var_list)
            self.assertEqual(current_lr, expected_lr)
        state_dict = sgd.state_dict()
        scheduler1 = paddle.optimizer.ReduceLROnPlateau(**kwargs)
        sgd1 = paddle.optimizer.SGD(learning_rate=scheduler1,
                                    parameter_list=linear.parameters())
        sgd1.set_dict(state_dict)
        self.assertEqual(scheduler.cooldown_counter,
                         scheduler1.cooldown_counter)
        self.assertEqual(scheduler.best.numpy()[0], scheduler1.best)
        self.assertEqual(scheduler.num_bad_epochs, scheduler1.num_bad_epochs)
        self.assertEqual(scheduler.last_epoch, scheduler1.last_epoch)
        self.assertEqual(scheduler.last_lr, scheduler1.last_lr)


def noam_lr(epoch_num, d_model, warmup_steps, learning_rate=1.0, verbose=False):
    if epoch_num == 0:
        a = 1
    else:
        a = math.pow(epoch_num, -0.5)
    b = math.pow(warmup_steps, -1.5) * epoch_num
    return learning_rate * math.pow(d_model, -0.5) * min(a, b)


def lambda_lr(epoch_num, learning_rate, lr_lambda, verbose=False):
    return learning_rate * lr_lambda(epoch_num)


def piecewise_lr(epoch_num, boundaries, values, verbose=False):
    assert len(boundaries) + 1 == len(values)
    for i in range(len(boundaries)):
        if epoch_num < boundaries[i]:
            return values[i]
    return values[len(values) - 1]


def exponential_lr(epoch_num, learning_rate, gamma, verbose=False):
    return learning_rate * gamma**epoch_num


def natural_exp_lr(epoch_num, learning_rate, gamma, verbose=False):
    return learning_rate * math.exp(-1 * gamma * epoch_num)


def inverse_time_lr(epoch_num, learning_rate, gamma, verbose=False):
    return learning_rate / (1 + gamma * epoch_num)


def polynomial_lr(epoch_num,
                  learning_rate,
                  decay_steps,
                  end_lr=0.0001,
                  power=1.0,
                  cycle=False,
                  verbose=False):

    if cycle:
        div = math.ceil(epoch_num / float(decay_steps))
        if epoch_num == 0:
            div = 1
        decay_steps = decay_steps * div
    else:
        epoch_num = min(epoch_num, decay_steps)
    return (learning_rate - end_lr) * (
        (1 - float(epoch_num) / float(decay_steps))**power) + end_lr

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lr
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return self.last_lr + (self.base_lr - self.eta_min) * (1 - math.cos(
                math.pi / self.T_max)) / 2

        return (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / (
            1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) * (
                self.last_lr - self.eta_min) + self.eta_min


cosine_annealing_lr_current = None


def cosine_annealing_lr(epoch_num,
                        learning_rate,
                        T_max,
                        eta_min=0,
                        verbose=False):
    global cosine_annealing_lr_current
    if epoch_num == 0:
        cosine_annealing_lr_current = learning_rate
    elif (epoch_num - 1 - T_max) % (2 * T_max) == 0:
        cosine_annealing_lr_current = cosine_annealing_lr_current + (
            learning_rate - eta_min) * (1 - math.cos(math.pi / float(T_max))
                                        ) / 2
    else:
        cosine_annealing_lr_current = (1 + math.cos(
            math.pi * epoch_num / float(T_max))) / (1 + math.cos(math.pi * (
                epoch_num - 1) / float(T_max))) * (cosine_annealing_lr_current -
                                                   eta_min) + eta_min
    return cosine_annealing_lr_current


def linear_warmup_lr(epoch_num,
                     learning_rate,
                     warmup_steps,
                     start_lr,
                     end_lr,
                     verbose=False):
    if epoch_num < warmup_steps:
        return start_lr + (end_lr - start_lr) * (float(epoch_num) /
                                                 float(warmup_steps))
    else:
        return learning_rate


def multi_step_lr(epoch_num,
                  learning_rate,
                  milestones,
                  gamma=0.1,
                  verbose=False):
    for i in range(len(milestones)):
        if epoch_num < milestones[i]:
            return learning_rate * (gamma**i)
    return learning_rate * (gamma**len(milestones))


def step_lr(epoch_num, learning_rate, step_size, gamma=0.1, verbose=False):
    return learning_rate * math.pow(gamma, epoch_num // step_size)


class TestLRScheduler(unittest.TestCase):
    def _test_static(self, python_func, paddle_api, kwarg, place):
        main_prog = fluid.Program()
        start_prog = fluid.Program()
        with fluid.program_guard(main_prog, start_prog):
            x = fluid.data(name='x', shape=[3, 4, 5])
            y = fluid.data(name='y', shape=[3, 4, 5])
            z = fluid.layers.fc(x, 100)
            loss = fluid.layers.mean(z)
            scheduler = paddle_api(**kwarg)
            adam = fluid.optimizer.Adam(learning_rate=scheduler)
            adam.minimize(loss)
            lr_var = adam._global_learning_rate()
            test_prog = main_prog.clone()

        num = 0
        exe = fluid.Executor(place)
        exe.run(start_prog)
        for epoch in range(5):
            for batch_id in range(2):
                out = exe.run(
                    main_prog,
                    feed={
                        'x': np.random.randn(3, 4, 5).astype('float32'),
                        'y': np.random.randn(3, 4, 5).astype('float32')
                    },
                    fetch_list=lr_var.name)
            self.assertEqual(out, np.array(python_func(num, **kwarg)))
            scheduler.step()
            num += 1

        for epoch in range(5):
            for batch_id in range(2):
                out = exe.run(
                    test_prog,
                    feed={
                        'x': np.random.randn(3, 4, 5).astype('float32'),
                        'y': np.random.randn(3, 4, 5).astype('float32')
                    },
                    fetch_list=lr_var.name)
            self.assertEqual(out, np.array(python_func(num, **kwarg)))
            scheduler.step()
            num += 1

        if isinstance(place, fluid.CPUPlace):
            compiled_train_prog = fluid.CompiledProgram(
                main_prog).with_data_parallel(
                    loss_name=loss.name, places=fluid.cpu_places(4))
            for epoch in range(5):
                python_result = python_func(num, **kwarg)
                for batch_id in range(2):
                    _ = exe.run(
                        compiled_train_prog,
                        feed={
                            'x': np.random.randn(12, 4, 5).astype('float32'),
                            'y': np.random.randn(12, 4, 5).astype('float32')
                        },
                        fetch_list=lr_var.name)
                scopes = compiled_train_prog._executor.local_scopes()
                out = np.array(scopes[0].var(lr_var.name).get_tensor())
                self.assertEqual(out, np.array(python_result))
                out = np.array(scopes[1].var(lr_var.name).get_tensor())
                self.assertEqual(out, np.array(python_result))
                out = np.array(scopes[2].var(lr_var.name).get_tensor())
                self.assertEqual(out, np.array(python_result))
                out = np.array(scopes[3].var(lr_var.name).get_tensor())
                self.assertEqual(out, np.array(python_result))
                scheduler.step()
                num += 1

            compiled_test_prog = fluid.CompiledProgram(
                test_prog).with_data_parallel(
                    loss_name=loss.name,
                    share_vars_from=compiled_train_prog,
                    places=fluid.cpu_places(4))
            for epoch in range(5):
                python_result = python_func(num, **kwarg)
                for batch_id in range(2):
                    _ = exe.run(
                        compiled_test_prog,
                        feed={
                            'x': np.random.randn(12, 4, 5).astype('float32'),
                            'y': np.random.randn(12, 4, 5).astype('float32')
                        },
                        fetch_list=lr_var.name)
                scopes = compiled_test_prog._executor.local_scopes()
                out = np.array(scopes[0].var(lr_var.name).get_tensor())
                self.assertEqual(out, np.array(python_result))
                out = np.array(scopes[1].var(lr_var.name).get_tensor())
                self.assertEqual(out, np.array(python_result))
                out = np.array(scopes[2].var(lr_var.name).get_tensor())
                self.assertEqual(out, np.array(python_result))
                out = np.array(scopes[3].var(lr_var.name).get_tensor())
                self.assertEqual(out, np.array(python_result))
                scheduler.step()
                num += 1

    def _test_dygraph(self, python_func, paddle_api, kwarg, place):
        x = np.random.uniform(-1, 1, [10, 10]).astype("float32")
        linear = paddle.nn.Linear(10, 10)
        scheduler = paddle_api(**kwarg)
        sgd = paddle.optimizer.SGD(learning_rate=scheduler,
                                   parameter_list=linear.parameters())
        for epoch in range(20):
            for batch_id in range(2):
                x = paddle.to_tensor(x)
                out = linear(x)
                loss = paddle.reduce_mean(out)
                out.backward()
                sgd.minimize(loss)
                linear.clear_gradients()

            self.assertAlmostEqual(sgd.current_step_lr(),
                                   python_func(epoch, **kwarg))
            if paddle_api.__name__ != "CosineAnnealingLR":
                scheduler.step()
            else:
                scheduler.step(epoch + 1)

    def test_scheduler(self):
        with self.assertRaises(NotImplementedError):
            paddle.optimizer.lr_scheduler._LRScheduler().step()
        with self.assertRaises(TypeError):
            paddle.optimizer.MultiStepLR(
                learning_rate="test", milestones=[1, 2, 3])
        with self.assertRaises(TypeError):
            paddle.optimizer.MultiStepLR(learning_rate=0.5, milestones='test')
        with self.assertRaises(ValueError):
            paddle.optimizer.MultiStepLR(
                learning_rate=0.5, milestones=[3, 2, 1])
        with self.assertRaises(ValueError):
            paddle.optimizer.MultiStepLR(
                learning_rate=0.5, milestones=[1, 2, 3], gamma=2)

        func_api_kwargs = [(noam_lr, paddle.optimizer.NoamLR, {
            "d_model": 0.01,
            "warmup_steps": 100,
            "verbose": False
        }), (piecewise_lr, paddle.optimizer.PiecewiseLR, {
            "boundaries": [3, 6, 9, 15, 20],
            "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "verbose": False
        }), (natural_exp_lr, paddle.optimizer.NaturalExpLR, {
            "learning_rate": 0.5,
            "gamma": 0.1,
            "verbose": False
        }), (inverse_time_lr, paddle.optimizer.InverseTimeLR, {
            "learning_rate": 0.5,
            "gamma": 0.1,
            "verbose": True
        }), (polynomial_lr, paddle.optimizer.PolynomialLR, {
            "learning_rate": 0.5,
            "decay_steps": 20,
            "end_lr": 0,
            "power": 1.0,
            "cycle": False,
            "verbose": False
        }), (polynomial_lr, paddle.optimizer.PolynomialLR, {
            "learning_rate": 0.5,
            "decay_steps": 20,
            "end_lr": 0,
            "power": 1.0,
            "cycle": True,
            "verbose": False
        }), (linear_warmup_lr, paddle.optimizer.LinearLrWarmup, {
            'learning_rate': 0.5,
            'warmup_steps': 20,
            'start_lr': 0,
            'end_lr': 0.5,
            "verbose": False
        }), (exponential_lr, paddle.optimizer.ExponentialLR, {
            "learning_rate": 0.5,
            "gamma": 0.9,
            "verbose": False
        }), (multi_step_lr, paddle.optimizer.MultiStepLR, {
            "learning_rate": 0.5,
            "milestones": [3, 6, 9, 15, 20],
            "gamma": 0.8,
            "verbose": True
        }), (step_lr, paddle.optimizer.StepLR, {
            "learning_rate": 0.5,
            "step_size": 2,
            "gamma": 0.8,
            "verbose": False
        }), (lambda_lr, paddle.optimizer.LambdaLR, {
            "learning_rate": 0.5,
            "lr_lambda": lambda x: 0.95**x,
            "verbose": False
        }), (cosine_annealing_lr, paddle.optimizer.CosineAnnealingLR, {
            "learning_rate": 0.5,
            "T_max": 10,
            "verbose": True
        })]

        for python_func, paddle_api, kwarg in func_api_kwargs:
            places = [fluid.CPUPlace()]
            if core.is_compiled_with_cuda():
                places.append(fluid.CUDAPlace(0))

            for place in places:
                paddle.enable_static()
                self._test_static(python_func, paddle_api, kwarg, place)
                paddle.disable_static(place)
                self._test_dygraph(python_func, paddle_api, kwarg, place)
                paddle.enable_static()


if __name__ == '__main__':
    unittest.main()
