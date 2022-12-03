# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

import paddle
import paddle.fluid as fluid


class TestModelAverage(unittest.TestCase):
    def test_model_average_static(self):
        paddle.enable_static()
        place = fluid.CPUPlace()
        shape = [2, 3, 8, 8]
        exe = fluid.Executor(place)
        train_program = fluid.Program()
        startup = fluid.Program()
        test_program = fluid.Program()
        with fluid.program_guard(train_program, startup):
            with fluid.unique_name.guard():
                data = fluid.data(name='X', shape=[None, 1], dtype='float32')
                hidden = fluid.layers.fc(input=data, size=10)
                loss = paddle.mean(hidden)
                test_program = train_program.clone()
                optimizer = paddle.optimizer.Momentum(
                    learning_rate=0.2, momentum=0.1
                )

                optimizer.minimize(loss)
                # build ModelAverage optimizer
                model_average = paddle.fluid.optimizer.ModelAverage(
                    0.15, min_average_window=2, max_average_window=10
                )

        exe.run(startup)
        for i in range(10):
            x = np.random.random(size=(10, 1)).astype('float32')
            (
                latest_b,
                sum_1,
                sum_2,
                sum_3,
                num_accumulates,
                old_num_accumulates,
                num_updates,
            ) = exe.run(
                program=train_program,
                feed={'X': x},
                fetch_list=[
                    'fc_0.b_0',
                    'fc_0.b_0_sum_1_0',
                    'fc_0.b_0_sum_2_0',
                    'fc_0.b_0_sum_3_0',
                    'fc_0.b_0_num_accumulates_0',
                    'fc_0.b_0_old_num_accumulates_0',
                    'fc_0.b_0_num_updates_0',
                ],
            )
        self.assertTrue(
            np.equal(sum_1, np.zeros(shape=[10], dtype='float32')).all()
        )
        self.assertTrue(
            np.equal(sum_2, np.zeros(shape=[10], dtype='float32')).all()
        )
        self.assertTrue(
            np.equal(num_accumulates, np.array([0], dtype='int64')).all()
        )
        self.assertTrue(
            np.equal(old_num_accumulates, np.array([2], dtype='int64')).all()
        )
        self.assertTrue(
            np.equal(num_updates, np.array([10], dtype='int64')).all()
        )

        average_b = (sum_1 + sum_2 + sum_3) / (
            num_accumulates + old_num_accumulates
        )
        # apply ModelAverage
        with model_average.apply(exe):
            x = np.random.random(size=(10, 1)).astype('float32')
            outs, b = exe.run(
                program=test_program,
                feed={'X': x},
                fetch_list=[loss.name, 'fc_0.b_0'],
            )
            self.assertAlmostEqual(np.mean(average_b), np.mean(b))

        x = np.random.random(size=(10, 1)).astype('float32')
        outs, b = exe.run(
            program=test_program,
            feed={'X': x},
            fetch_list=[loss.name, 'fc_0.b_0'],
        )
        self.assertAlmostEqual(np.mean(latest_b), np.mean(b))


if __name__ == "__main__":
    unittest.main()
