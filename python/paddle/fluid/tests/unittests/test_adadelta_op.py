#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest

import paddle
import paddle.fluid as fluid


class TestAdadeltaOp1(OpTest):
    def setUp(self):
        self.op_type = "adadelta"
        param = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        grad = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        # The squared gradient is positive
        avg_squared_grad = np.random.random((102, 105)).astype("float32")
        # The squared update is positive
        avg_squared_update = np.random.random((102, 105)).astype("float32")

        rho = 0.95
        epsilon = 1e-6

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'AvgSquaredGrad': avg_squared_grad,
            'AvgSquaredUpdate': avg_squared_update,
        }

        self.attrs = {'rho': rho, 'epsilon': epsilon}

        avg_squared_grad_out = rho * avg_squared_grad + (1 - rho) * np.square(
            grad
        )
        update = -np.multiply(
            np.sqrt(
                np.divide(
                    avg_squared_update + epsilon, avg_squared_grad_out + epsilon
                )
            ),
            grad,
        )

        avg_squared_update_out = rho * avg_squared_update + (
            1 - rho
        ) * np.square(update)

        param_out = param + update

        self.outputs = {
            'ParamOut': param_out,
            'AvgSquaredGradOut': avg_squared_grad_out,
            'AvgSquaredUpdateOut': avg_squared_update_out,
        }

    def test_check_output(self):
        self.check_output()


class TestAdadeltaOp2(OpTest):
    '''Test Adadelta op with default attribute values'''

    def setUp(self):
        self.op_type = "adadelta"
        param = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        grad = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        # The squared gradient is positive
        avg_squared_grad = np.random.random((102, 105)).astype("float32")
        # The squared update is positive
        avg_squared_update = np.random.random((102, 105)).astype("float32")

        rho = 0.95
        epsilon = 1e-6

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'AvgSquaredGrad': avg_squared_grad,
            'AvgSquaredUpdate': avg_squared_update,
        }

        avg_squared_grad_out = rho * avg_squared_grad + (1 - rho) * np.square(
            grad
        )
        update = -np.multiply(
            np.sqrt(
                np.divide(
                    avg_squared_update + epsilon, avg_squared_grad_out + epsilon
                )
            ),
            grad,
        )

        avg_squared_update_out = rho * avg_squared_update + (
            1 - rho
        ) * np.square(update)

        param_out = param + update

        self.outputs = {
            'ParamOut': param_out,
            'AvgSquaredGradOut': avg_squared_grad_out,
            'AvgSquaredUpdateOut': avg_squared_update_out,
        }

    def test_check_output(self):
        self.check_output()


class TestAdadeltaV2(unittest.TestCase):
    def test_adadelta_dygraph(self):
        paddle.disable_static(paddle.CPUPlace())
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear = paddle.nn.Linear(13, 5)
        # This can be any optimizer supported by dygraph.
        adam = paddle.optimizer.Adadelta(
            learning_rate=0.01,
            parameters=linear.parameters(),
            weight_decay=0.01,
        )
        out = linear(a)
        out.backward()
        adam.step()
        adam.clear_gradients()

    def test_adadelta(self):
        paddle.enable_static()
        place = fluid.CPUPlace()
        main = fluid.Program()
        with fluid.program_guard(main):
            x = fluid.layers.data(name='x', shape=[13], dtype='float32')
            y = fluid.layers.data(name='y', shape=[1], dtype='float32')
            y_predict = fluid.layers.fc(input=x, size=1, act=None)
            cost = paddle.nn.functional.square_error_cost(
                input=y_predict, label=y
            )
            avg_cost = paddle.mean(cost)

            rms_optimizer = paddle.optimizer.Adadelta(learning_rate=0.1)
            rms_optimizer.minimize(avg_cost)

            fetch_list = [avg_cost]
            train_reader = paddle.batch(
                paddle.dataset.uci_housing.train(), batch_size=1
            )
            feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for data in train_reader():
                exe.run(main, feed=feeder.feed(data), fetch_list=fetch_list)

    def test_raise_error(self):
        self.assertRaises(ValueError, paddle.optimizer.Adadelta, None)
        self.assertRaises(
            ValueError, paddle.optimizer.Adadelta, learning_rate=0.1, rho=None
        )
        self.assertRaises(
            ValueError,
            paddle.optimizer.Adadelta,
            learning_rate=0.1,
            epsilon=None,
        )


class TestAdadeltaV2Group(TestAdadeltaV2):
    def test_adadelta_dygraph(self):
        paddle.disable_static(paddle.CPUPlace())
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear_1 = paddle.nn.Linear(13, 5)
        linear_2 = paddle.nn.Linear(5, 5)
        # This can be any optimizer supported by dygraph.
        adam = paddle.optimizer.Adadelta(
            learning_rate=0.01,
            parameters=[
                {'params': linear_1.parameters()},
                {
                    'params': linear_2.parameters(),
                    'weight_decay': 0.001,
                },
            ],
            weight_decay=0.1,
        )
        out = linear_1(a)
        out = linear_2(out)
        out.backward()
        adam.step()
        adam.clear_gradients()


if __name__ == "__main__":
    unittest.main()
