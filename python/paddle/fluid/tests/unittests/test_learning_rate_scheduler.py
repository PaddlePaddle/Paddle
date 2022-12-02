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

import copy
import math
import unittest

import numpy as np

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.framework as framework
import paddle.fluid.layers as layers


def exponential_decay(
    learning_rate, global_step, decay_steps, decay_rate, staircase=False
):
    exponent = global_step / decay_steps
    if staircase:
        exponent = math.floor(exponent)
    return learning_rate * decay_rate**exponent


def natural_exp_decay(
    learning_rate, global_step, decay_steps, decay_rate, staircase=False
):
    exponent = float(global_step) / float(decay_steps)
    if staircase:
        exponent = math.floor(exponent)
    return learning_rate * math.exp(-1 * decay_rate * exponent)


def inverse_time_decay(
    learning_rate, global_step, decay_steps, decay_rate, staircase=False
):
    temp = float(global_step) / float(decay_steps)
    if staircase:
        temp = math.floor(temp)
    return learning_rate / (1 + decay_rate * temp)


def polynomial_decay(
    learning_rate,
    global_step,
    decay_steps,
    end_learning_rate=0.0001,
    power=1.0,
    cycle=False,
):
    if cycle:
        div = math.ceil(global_step / float(decay_steps))
        if div == 0:
            div = 1
        decay_steps = decay_steps * div
    else:
        global_step = min(global_step, decay_steps)
    return (learning_rate - end_learning_rate) * (
        (1 - float(global_step) / float(decay_steps)) ** power
    ) + end_learning_rate


def piecewise_decay(global_step, boundaries, values):
    assert len(boundaries) + 1 == len(values)
    for i in range(len(boundaries)):
        if global_step < boundaries[i]:
            return values[i]
    return values[len(values) - 1]


def cosine_decay(global_step, learning_rate, step_each_epoch, epochs):
    cur_epoch = math.floor(global_step / step_each_epoch)
    decayed_lr = (
        learning_rate * 0.5 * (math.cos(cur_epoch * math.pi / epochs) + 1)
    )
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
            linear = paddle.nn.Linear(10, 10)
            input = fluid.dygraph.to_variable(x)

            Exponential_scheduler = fluid.dygraph.ExponentialDecay(
                learning_rate=0.1,
                decay_steps=10000,
                decay_rate=0.5,
                staircase=True,
            )
            Step_scheduler = fluid.dygraph.StepDecay(0.5, step_size=3)
            Reducelr_scheduler = fluid.dygraph.ReduceLROnPlateau(
                learning_rate=1.0, decay_rate=0.5, patience=5, cooldown=3
            )

            adam1 = fluid.optimizer.Adam(
                learning_rate=Exponential_scheduler,
                parameter_list=linear.parameters(),
            )
            adam2 = fluid.optimizer.Adam(
                learning_rate=Step_scheduler, parameter_list=linear.parameters()
            )
            adam3 = fluid.optimizer.Adam(
                learning_rate=Reducelr_scheduler,
                parameter_list=linear.parameters(),
            )
            print(adam3.state_dict())

            for epoch in range(10):
                out = linear(input)
                loss = paddle.mean(out)
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
                staircase=True,
            )
            Step_scheduler_test = fluid.dygraph.StepDecay(0.5, step_size=3)
            Reducelr_scheduler_test = fluid.dygraph.ReduceLROnPlateau(
                learning_rate=1.0, decay_rate=0.5, patience=5, cooldown=3
            )

            fluid.dygraph.save_dygraph(adam1.state_dict(), "save_path")
            _, opt_state = fluid.dygraph.load_dygraph("save_path")
            adam_test = fluid.optimizer.Adam(
                learning_rate=Exponential_scheduler_test,
                parameter_list=linear.parameters(),
            )
            adam_test.set_dict(opt_state)
            self.assertEqual(
                adam_test._learning_rate.step_num,
                adam1._learning_rate.step_num,
                "epoch_num is different before and after set_dict",
            )

            fluid.dygraph.save_dygraph(adam2.state_dict(), "save_path")
            _, opt_state = fluid.dygraph.load_dygraph("save_path")
            adam_test = fluid.optimizer.Adam(
                learning_rate=Step_scheduler_test,
                parameter_list=linear.parameters(),
            )
            adam_test.set_dict(opt_state)
            self.assertEqual(
                adam_test._learning_rate.epoch_num,
                adam2._learning_rate.epoch_num,
                "epoch_num is different before and after set_dict",
            )
            self.assertEqual(
                adam_test._learning_rate(),
                adam2._learning_rate(),
                "current learning rate is different before and after set_dict",
            )

            fluid.dygraph.save_dygraph(adam3.state_dict(), "save_path")
            _, opt_state = fluid.dygraph.load_dygraph("save_path")
            adam_test = fluid.optimizer.Adam(
                learning_rate=Reducelr_scheduler_test,
                parameter_list=linear.parameters(),
            )
            adam_test.set_dict(opt_state)
            self.assertEqual(
                adam_test._learning_rate.best_loss,
                adam3._learning_rate.best_loss.numpy()[0],
                "best_loss is different before and after set_dict",
            )
            self.assertEqual(
                adam_test._learning_rate.cooldown_counter,
                adam3._learning_rate.cooldown_counter,
                "cooldown_counter is different before and after set_dict",
            )
            self.assertEqual(
                adam_test._learning_rate.num_bad_epochs,
                adam3._learning_rate.num_bad_epochs,
                "num_bad_epochs is different before and after set_dict",
            )
            self.assertEqual(
                adam_test._learning_rate.epoch_num,
                adam3._learning_rate.epoch_num,
                "epoch is different before and after set_dict",
            )
            self.assertEqual(
                adam_test._learning_rate(),
                adam3._learning_rate(),
                "current learning rate is different before and after set_dict",
            )

    def test_NoamDecay(self):
        with fluid.dygraph.guard():
            d_model = 0.01
            warmup_steps = 200
            learning_rate = 2.0
            lr = fluid.layers.noam_decay(d_model, warmup_steps, learning_rate)
            for step in range(5):
                step += 1
                right_result = noam_decay(
                    step, d_model, warmup_steps, learning_rate
                )
                fluid_result = lr()

                self.assertAlmostEqual(
                    right_result,
                    fluid_result[0],
                    msg='Failed lr scheduler in step {0}, Python result is {1}, Fluid result is {2}'.format(
                        step, right_result, fluid_result[0]
                    ),
                )

    def test_LinearLrWarmup(self):
        with fluid.dygraph.guard():
            lr = fluid.layers.polynomial_decay(
                learning_rate=1.0,
                decay_steps=10,
                end_learning_rate=0.0,
                power=1.0,
            )
            lr = fluid.layers.linear_lr_warmup(
                learning_rate=lr, warmup_steps=2, start_lr=0.0, end_lr=1.0
            )

            right_result = [0.5, 0.9, 0.8, 0.7, 0.6]
            for i in range(5):

                t = lr()

                np.testing.assert_allclose(
                    t.numpy()[0].item(), right_result[i], rtol=1e-05
                )

            with self.assertRaises(TypeError):
                lr = fluid.layers.linear_lr_warmup(
                    learning_rate="fake_lr",
                    warmup_steps=2,
                    start_lr=0.0,
                    end_lr=1.0,
                )

    def test_MultiStepDecay(self):
        with fluid.dygraph.guard():
            learning_rate = 0.5
            milestones = [2, 4, 8]
            decay_rate = 0.2
            linear = paddle.nn.Linear(10, 10)

            scheduler = fluid.dygraph.MultiStepDecay(
                learning_rate, milestones, decay_rate
            )

            adam = fluid.optimizer.AdamOptimizer(
                learning_rate=scheduler, parameter_list=linear.parameters()
            )
            for epoch in range(10):
                right_result = multi_step_decay(
                    epoch, learning_rate, milestones, decay_rate
                )
                fluid_result = adam.current_step_lr()
                scheduler.epoch()
                self.assertAlmostEqual(
                    right_result,
                    fluid_result,
                    msg='Failed lr scheduler in epoch {0}, Python result is {1}, Fluid result is {2}'.format(
                        epoch, right_result, fluid_result
                    ),
                )

            with self.assertRaises(ValueError):
                lr = fluid.dygraph.MultiStepDecay(
                    learning_rate, [30, 50, 20], 0.1
                )

            with self.assertRaises(ValueError):
                lr = fluid.dygraph.MultiStepDecay(
                    learning_rate, [20, 30, 50], 1
                )

            with self.assertRaises(TypeError):
                lr = fluid.dygraph.MultiStepDecay("test", [20, 30, 50])

            with self.assertRaises(ValueError):
                lr = fluid.dygraph.MultiStepDecay(-1, [20, 30, 50])

    def test_StepDecay(self):
        with fluid.dygraph.guard():
            learning_rate = 0.5
            step_size = 3
            decay_rate = 0.2
            scheduler = fluid.dygraph.StepDecay(
                learning_rate, step_size, decay_rate
            )
            for epoch in range(10):
                right_result = step_decay(
                    epoch, learning_rate, step_size, decay_rate
                )
                fluid_result = scheduler().numpy()[0]
                scheduler.epoch()
                self.assertAlmostEqual(
                    right_result,
                    fluid_result,
                    msg='Failed lr scheduler in epoch {0}, Python result is {1}, Fluid result is {2}'.format(
                        epoch, right_result, fluid_result
                    ),
                )

            with self.assertRaises(TypeError):
                lr = fluid.dygraph.StepDecay(learning_rate, "test", 0.1)

            with self.assertRaises(ValueError):
                lr = fluid.dygraph.StepDecay(learning_rate, 20, 2)

    def test_LambdaDecay(self):
        with fluid.dygraph.guard():
            learning_rate = 0.5
            lr_lambda = lambda x: 0.95**x
            scheduler = fluid.dygraph.LambdaDecay(learning_rate, lr_lambda)

            linear = paddle.nn.Linear(10, 10)
            adam = fluid.optimizer.Adam(
                scheduler, parameter_list=linear.parameters()
            )

            for epoch in range(30):
                right_result = lambda_decay(epoch, learning_rate, lr_lambda)
                fluid_result = scheduler().numpy()[0]
                scheduler.epoch()
                self.assertAlmostEqual(
                    right_result,
                    fluid_result,
                    msg='Failed lr scheduler in epoch {0}, Python result is {1}, Fluid result is {2}'.format(
                        epoch, right_result, fluid_result
                    ),
                )

            with self.assertRaises(TypeError):
                lr = fluid.dygraph.LambdaDecay(learning_rate, "test")


class TestLearningRateDecay(unittest.TestCase):
    def check_decay(self, python_decay_fn, fluid_decay_fn, kwargs):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for place in places:
            self.check_decay_with_place(
                place, python_decay_fn, fluid_decay_fn, kwargs
            )

    def check_decay_with_place(
        self, place, python_decay_fn, fluid_decay_fn, kwargs
    ):
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
            (lr_val,) = exe.run(main_prog, feed={}, fetch_list=[decayed_lr])
            python_decayed_lr = python_decay_fn(
                global_step=float(step), **kwargs
            )
            self.assertAlmostEqual(
                python_decayed_lr,
                lr_val[0],
                msg='Failed lr scheduler is {0}, step {1}, Python result is {2}, Fluid result is {3}'.format(
                    python_decay_fn.__name__,
                    str(step),
                    str(python_decayed_lr),
                    str(lr_val[0]),
                ),
            )

    def test_decay(self):
        common_kwargs_true = {
            "learning_rate": 1.0,
            "decay_steps": 5,
            "decay_rate": 0.5,
            "staircase": True,
        }
        common_kwargs_false = copy.deepcopy(common_kwargs_true)
        common_kwargs_false["staircase"] = False

        decay_fns = [
            (exponential_decay, layers.exponential_decay, common_kwargs_true),
            (exponential_decay, layers.exponential_decay, common_kwargs_false),
            (natural_exp_decay, layers.natural_exp_decay, common_kwargs_true),
            (natural_exp_decay, layers.natural_exp_decay, common_kwargs_false),
            (inverse_time_decay, layers.inverse_time_decay, common_kwargs_true),
            (
                inverse_time_decay,
                layers.inverse_time_decay,
                common_kwargs_false,
            ),
            (
                polynomial_decay,
                layers.polynomial_decay,
                {"learning_rate": 1.0, "decay_steps": 5, "cycle": True},
            ),
            (
                polynomial_decay,
                layers.polynomial_decay,
                {"learning_rate": 1.0, "decay_steps": 5, "cycle": False},
            ),
            (
                piecewise_decay,
                layers.piecewise_decay,
                {"boundaries": [3, 6, 9], "values": [0.1, 0.2, 0.3, 0.4]},
            ),
            (
                cosine_decay,
                layers.cosine_decay,
                {"learning_rate": 0.1, "step_each_epoch": 100, "epochs": 120},
            ),
            (
                noam_decay,
                layers.noam_decay,
                {"d_model": 0.01, "warmup_steps": 200, "learning_rate": 2.0},
            ),
        ]

        for py_decay_fn, fluid_decay_fn, kwargs in decay_fns:
            print(
                "class="
                + self.__class__.__name__
                + " decay_fn="
                + py_decay_fn.__name__
                + " kwargs="
                + str(kwargs)
            )
            main_program = framework.Program()
            startup_program = framework.Program()
            with framework.program_guard(main_program, startup_program):
                self.check_decay(py_decay_fn, fluid_decay_fn, kwargs)


class TestLinearWamrupLearningRateDecay(unittest.TestCase):
    def check_decay_with_place(
        self, place, python_decay_fn, fluid_decay_fn, kwargs
    ):
        main_prog = fluid.Program()
        startup_prog = fluid.Program()

        warmup_steps = 10
        start_lr = 0.1 / 3.0
        end_lr = 0.1

        with fluid.program_guard(main_prog, startup_prog):
            decayed_lr = layers.linear_lr_warmup(
                fluid_decay_fn(**kwargs), warmup_steps, start_lr, end_lr
            )

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_prog)

        for step in range(20):
            # Step of NoamDecay starts from 1.
            if fluid_decay_fn.__name__ == 'noam_decay':
                step += 1
            (lr_val,) = exe.run(main_prog, feed={}, fetch_list=[decayed_lr])
            if step < warmup_steps:
                python_decayed_lr = linear_lr_warmup(
                    float(step), warmup_steps, start_lr, end_lr
                )
            else:
                python_decayed_lr = python_decay_fn(
                    global_step=float(step), **kwargs
                )
            self.assertAlmostEqual(
                python_decayed_lr,
                lr_val[0],
                msg='Test {0} Failed, step {1}, Python result is {2}, Fluid result is {3}'.format(
                    python_decay_fn.__name__,
                    str(step),
                    str(python_decayed_lr),
                    str(lr_val[0]),
                ),
            )


class TestLinearWamrupLearningRateDecayWithScalarInput(unittest.TestCase):
    def run_scalar_lr(self, place, lr, start_lr, end_lr):
        main_prog = fluid.Program()
        startup_prog = fluid.Program()

        warmup_steps = 10

        with fluid.program_guard(main_prog, startup_prog):
            decayed_lr = layers.linear_lr_warmup(
                lr, warmup_steps, start_lr, end_lr
            )

        exe = fluid.Executor(place)
        exe.run(startup_prog)

        for step in range(20):
            (lr_val,) = exe.run(main_prog, feed={}, fetch_list=[decayed_lr])
            if step < warmup_steps:
                expected_lr = linear_lr_warmup(
                    float(step), warmup_steps, start_lr, end_lr
                )
            else:
                expected_lr = lr
            self.assertAlmostEqual(
                expected_lr,
                lr_val[0],
                msg='Test failed, step {0}, expected {1}, but got {2}'.format(
                    step, expected_lr, lr_val[0]
                ),
            )

    def test_scalar_lr(self):
        def run_places(lr, start_lr, end_lr):
            places = [fluid.CPUPlace()]
            if core.is_compiled_with_cuda():
                places.append(fluid.CUDAPlace(0))
            for p in places:
                self.run_scalar_lr(p, lr, start_lr, end_lr)

        # float
        lr = 0.2
        start_lr = 0.1 / 3.0
        end_lr = 0.2
        run_places(lr, start_lr, end_lr)

        # int end_lr
        lr = 2.0
        start_lr = 0.1 / 3.0
        end_lr = 1
        run_places(lr, start_lr, end_lr)

        # int
        lr = 1
        start_lr = 0
        end_lr = 1
        run_places(lr, start_lr, end_lr)


if __name__ == '__main__':
    unittest.main()
