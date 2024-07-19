# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import unittest

import numpy as np

import paddle
from paddle.base import core


def reduce_lr_on_plateau(
    decay_rate, threshold, cooldown, patience, m, n, loss, var_list
):
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


class TestReduceOnPlateauDecay(unittest.TestCase):
    def test_ReduceLR(self):
        # the decay rate must be less than 1.0
        with self.assertRaises(ValueError):
            paddle.optimizer.lr.ReduceOnPlateau(learning_rate=1.0, factor=2.0)
        # the mode must be "min" or "max"
        with self.assertRaises(ValueError):
            paddle.optimizer.lr.ReduceOnPlateau(learning_rate=1.0, mode="test")
        # the threshold_mode must be "rel" or "abs"
        with self.assertRaises(ValueError):
            paddle.optimizer.lr.ReduceOnPlateau(
                learning_rate=1.0, threshold_mode="test"
            )
        with self.assertRaises(TypeError):
            paddle.optimizer.lr.ReduceOnPlateau(learning_rate="test")
        with self.assertRaises(TypeError):
            paddle.optimizer.lr.ReduceOnPlateau(learning_rate=0.5).step("test")

        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))

        for place in places:
            for m, n in zip(
                ['min', 'max', 'min', 'max'], ['rel', 'rel', 'abs', 'abs']
            ):
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

    def _test_static(self, place, kwargs):
        paddle.enable_static()

        best = float("-10000") if kwargs['mode'] == "max" else float("10000")
        current_lr = 1.0
        cooldown_counter = 0
        num_bad_epochs = 0
        var_list = [best, current_lr, cooldown_counter, num_bad_epochs]

        main_prog = paddle.static.Program()
        start_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, start_prog):
            x = paddle.pir.core.create_parameter(
                'float32',
                [1],
                name='x',
                initializer=paddle.nn.initializer.ConstantInitializer(
                    value=float(1), force_cpu=False
                ),
            )
            paddle.increment(x)
            loss = paddle.sin(x)
            scheduler = paddle.optimizer.lr.ReduceOnPlateau(**kwargs)
            adam = paddle.optimizer.Adam(learning_rate=scheduler)
            adam.minimize(loss)
            lr_var = adam._global_learning_rate()
            # test_prog = main_prog.clone()

        exe = paddle.static.Executor(place)
        exe.run(start_prog)

        for epoch in range(20):
            for batch_id in range(1):
                out, actual_lr = exe.run(main_prog, fetch_list=[loss, lr_var])
                expected_lr = reduce_lr_on_plateau(
                    kwargs['factor'],
                    kwargs['threshold'],
                    kwargs['cooldown'],
                    kwargs['patience'],
                    kwargs['mode'],
                    kwargs['threshold_mode'],
                    out[0],
                    var_list,
                )

            scheduler.step(out[0])
            actual_lr = scheduler()
            self.assertEqual(actual_lr, np.array(expected_lr))


if __name__ == '__main__':
    unittest.main()
