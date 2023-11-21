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
import unittest

import numpy as np

import paddle

paddle.enable_static()


class TestBuildModule(unittest.TestCase):
    def test_basic_network(self):
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            x = paddle.static.data('x', [4, 4], dtype='float32')
            y = paddle.static.data('y', [4, 4], dtype='float32')
            divide_out = paddle.divide(x, y)
            sum_out = paddle.sum(divide_out)
            exe = paddle.static.Executor()
            x_feed = np.ones([4, 4], dtype=np.float32) * 10
            y_feed = np.ones([4, 4], dtype=np.float32) * 2
        (sum_value,) = exe.run(
            main_program,
            feed={'x': x_feed, 'y': y_feed},
            fetch_list=[sum_out],
        )
        self.assertEqual(sum_value, 5 * 4 * 4)

        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            x = paddle.static.data('x', [4, 4], dtype='float32')
            out = paddle.mean(x)
            exe = paddle.static.Executor()
            x_feed = np.ones([4, 4], dtype=np.float32) * 10
            (sum_value,) = exe.run(feed={'x': x_feed}, fetch_list=[out])
            self.assertEqual(sum_value, 10)

    def test_basic_network_without_guard(self):
        x = paddle.static.data('x', [4, 4], dtype='float32')
        y = paddle.static.data('y', [4, 4], dtype='float32')
        divide_out = paddle.divide(x, y)
        sum_out = paddle.sum(divide_out)
        exe = paddle.static.Executor()
        x_feed = np.ones([4, 4], dtype=np.float32) * 10
        y_feed = np.ones([4, 4], dtype=np.float32) * 2
        (sum_value,) = exe.run(
            feed={'x': x_feed, 'y': y_feed},
            fetch_list=[sum_out],
        )
        self.assertEqual(sum_value, 5 * 4 * 4)

        out = paddle.mean(x)
        exe = paddle.static.Executor()
        x_feed = np.ones([4, 4], dtype=np.float32) * 10
        (sum_value,) = exe.run(
            feed={'x': x_feed, 'y': y_feed}, fetch_list=[out]
        )
        self.assertEqual(sum_value, 10)

    def test_train_network(self):
        x_data = np.array(
            [[1.0], [3.0], [5.0], [9.0], [10.0], [20.0]], dtype="float32"
        )
        y_data = np.array(
            [[12.0], [16.0], [20.0], [28.0], [30.0], [50.0]], dtype="float32"
        )
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data(name="x", shape=[6, 1], dtype="float32")
            y = paddle.static.data(name="y", shape=[6, 1], dtype="float32")
            linear = paddle.nn.Linear(in_features=1, out_features=1)
            mse_loss = paddle.nn.MSELoss()
            sgd_optimizer = paddle.optimizer.SGD(
                learning_rate=0.001, parameters=linear.parameters()
            )
            exe = paddle.static.Executor()

            y_predict = linear(x)
            loss = mse_loss(y_predict, y)
            sgd_optimizer.minimize(loss)

            exe.run(startup_program)
            total_epoch = 5000
            for i in range(total_epoch):
                (loss_value,) = exe.run(
                    feed={'x': x_data, 'y': y_data}, fetch_list=[loss]
                )

            print(f"loss is {loss_value} after {total_epoch} iteration")

            self.assertLess(loss_value, 0.1)


if __name__ == "__main__":
    unittest.main()
