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
import os
import unittest

import numpy as np

import paddle
from paddle import base
from paddle.base import core, framework


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
        with base.dygraph.guard():
            x = np.random.uniform(-1, 1, [3, 10]).astype("float32")
            linear = paddle.nn.Linear(10, 10)
            input = paddle.to_tensor(x)

            Exponential_scheduler = paddle.optimizer.lr.ExponentialDecay(
                learning_rate=0.1,
                gamma=0.5,
            )
            Step_scheduler = paddle.optimizer.lr.StepDecay(0.5, step_size=3)
            Reducelr_scheduler = paddle.optimizer.lr.ReduceOnPlateau(
                learning_rate=1.0, factor=0.5, patience=5, cooldown=3
            )

            adam1 = paddle.optimizer.Adam(
                learning_rate=Exponential_scheduler,
                parameters=linear.parameters(),
            )
            adam2 = paddle.optimizer.Adam(
                learning_rate=Step_scheduler, parameters=linear.parameters()
            )
            adam3 = paddle.optimizer.Adam(
                learning_rate=Reducelr_scheduler,
                parameters=linear.parameters(),
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

                Step_scheduler.get_lr()
                Reducelr_scheduler.step(loss)

            paddle.save(linear.state_dict(), "save_path.pdparams")

            Exponential_scheduler_test = paddle.optimizer.lr.ExponentialDecay(
                learning_rate=0.1,
                gamma=0.5,
            )
            Step_scheduler_test = paddle.optimizer.lr.StepDecay(
                0.5, step_size=3
            )
            Reducelr_scheduler_test = paddle.optimizer.lr.ReduceOnPlateau(
                learning_rate=1.0, factor=0.5, patience=5, cooldown=3
            )

            paddle.save(adam1.state_dict(), "save_path.pdopt")
            opt_state = paddle.load("save_path.pdopt")
            adam_test = paddle.optimizer.Adam(
                learning_rate=Exponential_scheduler_test,
                parameters=linear.parameters(),
            )
            adam_test.set_state_dict(opt_state)
            self.assertEqual(
                adam_test._learning_rate.last_epoch,
                adam1._learning_rate.last_epoch,
                "last_epoch is different before and after set_state_dict",
            )

            paddle.save(adam2.state_dict(), "save_path.pdopt")
            opt_state = paddle.load("save_path.pdopt")
            adam_test = paddle.optimizer.Adam(
                learning_rate=Step_scheduler_test,
                parameters=linear.parameters(),
            )
            adam_test.set_state_dict(opt_state)
            self.assertEqual(
                adam_test._learning_rate.last_epoch,
                adam2._learning_rate.last_epoch,
                "epoch_num is different before and after set_state_dict",
            )
            self.assertEqual(
                adam_test._learning_rate(),
                adam2._learning_rate(),
                "current learning rate is different before and after set_state_dict",
            )

            paddle.save(adam3.state_dict(), "save_path.pdopt")
            opt_state = paddle.load("save_path.pdopt")
            adam_test = paddle.optimizer.Adam(
                learning_rate=Reducelr_scheduler_test,
                parameters=linear.parameters(),
            )
            adam_test.set_state_dict(opt_state)
            self.assertEqual(
                adam_test._learning_rate.best,
                adam3._learning_rate.best,
                "best_loss is different before and after set_state_dict",
            )
            self.assertEqual(
                adam_test._learning_rate.cooldown_counter,
                adam3._learning_rate.cooldown_counter,
                "cooldown_counter is different before and after set_state_dict",
            )
            self.assertEqual(
                adam_test._learning_rate.num_bad_epochs,
                adam3._learning_rate.num_bad_epochs,
                "num_bad_epochs is different before and after set_state_dict",
            )
            self.assertEqual(
                adam_test._learning_rate.last_epoch,
                adam3._learning_rate.last_epoch,
                "epoch is different before and after set_state_dict",
            )
            self.assertEqual(
                adam_test._learning_rate(),
                adam3._learning_rate(),
                "current learning rate is different before and after set_state_dict",
            )

    def test_NoamDecay(self):
        with base.dygraph.guard():
            d_model = 0.01
            warmup_steps = 200
            learning_rate = 2.0
            lr = paddle.optimizer.lr.noam_decay(
                d_model, warmup_steps, learning_rate
            )
            for step in range(5):
                step += 1
                right_result = noam_decay(
                    step, d_model, warmup_steps, learning_rate
                )
                lr.step()
                base_result = lr()

                self.assertAlmostEqual(
                    right_result,
                    base_result,
                    msg=f'Failed lr scheduler in step {step}, Python result is {right_result}, Fluid result is {base_result}',
                )

    def test_LinearLrWarmup(self):
        with base.dygraph.guard():
            lr = paddle.optimizer.lr.PolynomialDecay(
                learning_rate=1.0,
                decay_steps=10,
                end_lr=0.0,
                power=1.0,
            )
            lr.step()
            lr = paddle.optimizer.lr.LinearWarmup(
                learning_rate=lr, warmup_steps=2, start_lr=0.0, end_lr=1.0
            )
            lr.step()
            right_result = [0.5, 0.9, 0.8, 0.7, 0.6]
            for i in range(5):
                if i == 1:
                    lr.step()
                t = lr()
                lr.step()
                np.testing.assert_allclose(t, right_result[i], rtol=1e-05)

            with self.assertRaises(TypeError):
                lr = paddle.optimizer.lr.linear_lr_warmup(
                    learning_rate="fake_lr",
                    warmup_steps=2,
                    start_lr=0.0,
                    end_lr=1.0,
                )

    def test_MultiStepDecay(self):
        with base.dygraph.guard():
            learning_rate = 0.5
            milestones = [2, 4, 8]
            decay_rate = 0.2
            linear = paddle.nn.Linear(10, 10)

            scheduler = paddle.optimizer.lr.MultiStepDecay(
                learning_rate, milestones, decay_rate
            )

            adam = paddle.optimizer.Adam(
                learning_rate=scheduler, parameters=linear.parameters()
            )
            for epoch in range(10):
                right_result = multi_step_decay(
                    epoch, learning_rate, milestones, decay_rate
                )
                base_result = adam.get_lr()
                adam.step()
                scheduler.step()
                self.assertAlmostEqual(
                    right_result,
                    base_result,
                    msg=f'Failed lr scheduler in epoch {epoch}, Python result is {right_result}, Fluid result is {base_result}',
                )

            with self.assertRaises(ValueError):
                lr = paddle.optimizer.lr.MultiStepDecay(
                    learning_rate, [30, 50, 20], 0.1
                )

            with self.assertRaises(ValueError):
                lr = paddle.optimizer.lr.MultiStepDecay(
                    learning_rate, [20, 30, 50], 1
                )

            with self.assertRaises(TypeError):
                lr = paddle.optimizer.lr.MultiStepDecay("test", [20, 30, 50])

            with self.assertRaises(ValueError):
                lr = paddle.optimizer.lr.MultiStepDecay(-1, [20, 30, 50])

    def test_StepDecay(self):
        with base.dygraph.guard():
            learning_rate = 0.5
            step_size = 3
            decay_rate = 0.2
            scheduler = paddle.optimizer.lr.StepDecay(
                learning_rate, step_size, decay_rate
            )
            for epoch in range(10):
                right_result = step_decay(
                    epoch, learning_rate, step_size, decay_rate
                )
                base_result = scheduler()
                scheduler.get_lr()
                scheduler.step()
                self.assertAlmostEqual(
                    right_result,
                    base_result,
                    msg=f'Failed lr scheduler in epoch {epoch}, Python result is {right_result}, Fluid result is {base_result}',
                )

            with self.assertRaises(TypeError):
                lr = paddle.optimizer.lr.StepDecay(learning_rate, "test", 0.1)

            with self.assertRaises(ValueError):
                lr = paddle.optimizer.lr.StepDecay(learning_rate, 20, 2)

    def test_LambdaDecay(self):
        with base.dygraph.guard():
            learning_rate = 0.5
            lr_lambda = lambda x: 0.95**x
            scheduler = paddle.optimizer.lr.LambdaDecay(
                learning_rate, lr_lambda
            )

            linear = paddle.nn.Linear(10, 10)
            adam = paddle.optimizer.Adam(
                scheduler, parameters=linear.parameters()
            )

            for epoch in range(30):
                right_result = lambda_decay(epoch, learning_rate, lr_lambda)
                base_result = scheduler()
                scheduler.get_lr()
                scheduler.step()
                self.assertAlmostEqual(
                    right_result,
                    base_result,
                    msg=f'Failed lr scheduler in epoch {epoch}, Python result is {right_result}, Fluid result is {base_result}',
                )

            with self.assertRaises(TypeError):
                lr = paddle.optimizer.lr.LambdaDecay(learning_rate, "test")


class TestLearningRateDecay(unittest.TestCase):
    def check_decay(self, python_decay_fn, base_decay_fn, kwargs):
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for place in places:
            self.check_decay_with_place(
                place, python_decay_fn, base_decay_fn, kwargs
            )

    def check_decay_with_place(
        self, place, python_decay_fn, base_decay_fn, kwargs
    ):
        main_prog = base.Program()
        startup_prog = base.Program()

        with base.program_guard(main_prog, startup_prog):
            decayed_lr = base_decay_fn(**kwargs)

        place = base.CPUPlace()
        exe = base.Executor(place)

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
                places=6,
                msg=f'Failed lr scheduler is {python_decay_fn.__name__}, step {step}, Python result is {python_decayed_lr}, Fluid result is {lr_val[0]}',
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
            (
                exponential_decay,
                paddle.optimizer.lr.exponential_decay,
                common_kwargs_true,
            ),
            (
                exponential_decay,
                paddle.optimizer.lr.exponential_decay,
                common_kwargs_false,
            ),
            (
                natural_exp_decay,
                paddle.optimizer.lr.natural_exp_decay,
                common_kwargs_true,
            ),
            (
                natural_exp_decay,
                paddle.optimizer.lr.natural_exp_decay,
                common_kwargs_false,
            ),
            (
                inverse_time_decay,
                paddle.optimizer.lr.inverse_time_decay,
                common_kwargs_true,
            ),
            (
                inverse_time_decay,
                paddle.optimizer.lr.inverse_time_decay,
                common_kwargs_false,
            ),
            (
                polynomial_decay,
                paddle.optimizer.lr.polynomial_decay,
                {"learning_rate": 1.0, "decay_steps": 5, "cycle": True},
            ),
            (
                polynomial_decay,
                paddle.optimizer.lr.polynomial_decay,
                {"learning_rate": 1.0, "decay_steps": 5, "cycle": False},
            ),
            (
                piecewise_decay,
                paddle.optimizer.lr.piecewise_decay,
                {"boundaries": [3, 6, 9], "values": [0.1, 0.2, 0.3, 0.4]},
            ),
            (
                cosine_decay,
                paddle.optimizer.lr.cosine_decay,
                {"learning_rate": 0.1, "step_each_epoch": 100, "epochs": 120},
            ),
            (
                noam_decay,
                paddle.optimizer.lr.noam_decay,
                {"d_model": 0.01, "warmup_steps": 200, "learning_rate": 2.0},
            ),
        ]

        for py_decay_fn, base_decay_fn, kwargs in decay_fns:
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
                self.check_decay(py_decay_fn, base_decay_fn, kwargs)


class TestLinearWamrupLearningRateDecay(unittest.TestCase):
    def check_decay_with_place(
        self, place, python_decay_fn, base_decay_fn, kwargs
    ):
        main_prog = base.Program()
        startup_prog = base.Program()

        warmup_steps = 10
        start_lr = 0.1 / 3.0
        end_lr = 0.1

        with base.program_guard(main_prog, startup_prog):
            decayed_lr = paddle.optimizer.lr.linear_lr_warmup(
                base_decay_fn(**kwargs), warmup_steps, start_lr, end_lr
            )

        place = base.CPUPlace()
        exe = base.Executor(place)
        exe.run(startup_prog)

        for step in range(20):
            # Step of NoamDecay starts from 1.
            if base_decay_fn.__name__ == 'noam_decay':
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
                msg=f'Test {python_decay_fn.__name__} Failed, step {step}, Python result is {python_decayed_lr}, Fluid result is {lr_val[0]}',
            )


class TestLinearWamrupLearningRateDecayWithScalarInput(unittest.TestCase):
    def run_scalar_lr(self, place, lr, start_lr, end_lr):
        main_prog = base.Program()
        startup_prog = base.Program()

        warmup_steps = 10

        with base.program_guard(main_prog, startup_prog):
            decayed_lr = paddle.optimizer.lr.linear_lr_warmup(
                lr, warmup_steps, start_lr, end_lr
            )

        exe = base.Executor(place)
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
                places=6,
                msg=f'Test failed, step {step}, expected {expected_lr}, but got {lr_val[0]}',
            )

    def test_scalar_lr(self):
        def run_places(lr, start_lr, end_lr):
            places = []
            if (
                os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
                in ['1', 'true', 'on']
                or not core.is_compiled_with_cuda()
            ):
                places.append(base.CPUPlace())
            if core.is_compiled_with_cuda():
                places.append(base.CUDAPlace(0))
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
    paddle.enable_static()
    unittest.main()
