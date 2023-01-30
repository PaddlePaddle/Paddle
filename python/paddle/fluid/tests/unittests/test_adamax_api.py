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

<<<<<<< HEAD
import unittest

import numpy as np

import paddle
import paddle.fluid as fluid


class TestAdamaxAPI(unittest.TestCase):
    def test_adamax_api_dygraph(self):
=======
from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest
import paddle
import paddle.fluid as fluid
from paddle.fluid.framework import _test_eager_guard


class TestAdamaxAPI(unittest.TestCase):

    def func_adamax_api_dygraph(self):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        paddle.disable_static()
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear = paddle.nn.Linear(13, 5)
<<<<<<< HEAD
        adam = paddle.optimizer.Adamax(
            learning_rate=0.01,
            parameters=linear.parameters(),
            weight_decay=0.01,
        )
=======
        adam = paddle.optimizer.Adamax(learning_rate=0.01,
                                       parameters=linear.parameters(),
                                       weight_decay=0.01)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        out = linear(a)
        out.backward()
        adam.step()
        adam.clear_gradients()

<<<<<<< HEAD
    def test_adamax_api(self):
=======
    def test_adamax_api_dygraph(self):
        with _test_eager_guard():
            self.func_adamax_api_dygraph()
        self.func_adamax_api_dygraph()

    def func_adamax_api(self):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        paddle.enable_static()
        place = fluid.CPUPlace()
        shape = [2, 3, 8, 8]
        exe = fluid.Executor(place)
        train_prog = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(train_prog, startup):
            with fluid.unique_name.guard():
                data = fluid.data(name="data", shape=shape)
<<<<<<< HEAD
                conv = paddle.static.nn.conv2d(data, 8, 3)
                loss = paddle.mean(conv)
                beta1 = 0.85
                beta2 = 0.95
                opt = paddle.optimizer.Adamax(
                    learning_rate=1e-5,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=0.01,
                    epsilon=1e-8,
                )
=======
                conv = fluid.layers.conv2d(data, 8, 3)
                loss = paddle.mean(conv)
                beta1 = 0.85
                beta2 = 0.95
                opt = paddle.optimizer.Adamax(learning_rate=1e-5,
                                              beta1=beta1,
                                              beta2=beta2,
                                              weight_decay=0.01,
                                              epsilon=1e-8)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                opt.minimize(loss)

        exe.run(startup)
        data_np = np.random.random(shape).astype('float32')
        rets = exe.run(train_prog, feed={"data": data_np}, fetch_list=[loss])
        assert rets[0] is not None

<<<<<<< HEAD

class TestAdamaxAPIGroup(TestAdamaxAPI):
    def test_adamax_api_dygraph(self):
=======
    def test_adamax_api(self):
        with _test_eager_guard():
            self.func_adamax_api()
        self.func_adamax_api()


class TestAdamaxAPIGroup(TestAdamaxAPI):

    def func_adamax_api_dygraph(self):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        paddle.disable_static()
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear_1 = paddle.nn.Linear(13, 5)
        linear_2 = paddle.nn.Linear(5, 3)
        # This can be any optimizer supported by dygraph.
<<<<<<< HEAD
        adam = paddle.optimizer.Adamax(
            learning_rate=0.01,
            parameters=[
                {'params': linear_1.parameters()},
                {
                    'params': linear_2.parameters(),
                    'weight_decay': 0.001,
                    'beta1': 0.1,
                    'beta2': 0.99,
                },
            ],
            weight_decay=0.1,
        )
=======
        adam = paddle.optimizer.Adamax(learning_rate=0.01,
                                       parameters=[{
                                           'params':
                                           linear_1.parameters()
                                       }, {
                                           'params':
                                           linear_2.parameters(),
                                           'weight_decay':
                                           0.001,
                                           'beta1':
                                           0.1,
                                           'beta2':
                                           0.99
                                       }],
                                       weight_decay=0.1)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        out = linear_1(a)
        out = linear_2(out)
        out.backward()
        adam.step()
        adam.clear_gradients()

<<<<<<< HEAD
=======
    def test_adamax_api_dygraph(self):
        with _test_eager_guard():
            self.func_adamax_api_dygraph()
        self.func_adamax_api_dygraph()

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

if __name__ == "__main__":
    unittest.main()
