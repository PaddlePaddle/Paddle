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

from __future__ import print_function

import unittest
import sys
sys.path.append("..")
from op_test import OpTest
import numpy as np
import paddle.fluid.core as core
from paddle.fluid.op import Operator
import paddle.fluid as fluid
import paddle

paddle.enable_static()
SEED = 2021


class TestRMSProp(OpTest):
    def setUp(self, epsilon=1e-6):
        np.random.seed(5)  # fix seed
        self.op_type = "rmsprop"
        self.set_npu()
        self.place = paddle.NPUPlace(0)
        self.set_size()
        self.set_centered()

        self.param = np.random.random(self.size).astype("float32")

        self.mean_square = np.random.uniform(
            low=1, high=2, size=self.size).astype("float32")

        self.learning_rate = np.array([0.01]).astype("float32")

        self.grad = np.random.random(self.size).astype("float32")

        self.moment = np.random.uniform(
            low=0, high=1, size=self.size).astype("float32")

        self.epsilon = epsilon
        self.decay = 0.9
        self.momentum = 0.1

        self.inputs = {
            "Grad": self.grad,
            "Param": self.param,
            "MeanSquare": self.mean_square,
            "LearningRate": self.learning_rate,
            "Moment": self.moment
        }

        self.attrs = {
            "epsilon": self.epsilon,
            "decay": self.decay,
            "momentum": self.momentum,
            "centered": self.centered
        }

        self.ms_out = self.decay * self.mean_square + (1 - self.decay
                                                       ) * self.grad * self.grad
        self.mg_out = np.array([])
        if self.centered:
            self.mean_grad = np.random.random(self.size).astype("float32")
            self.mg_out = self.decay * self.mean_grad + (1 - self.decay
                                                         ) * self.grad
            self.moment_out = self.momentum * self.moment + \
                              self.learning_rate * self.grad / np.sqrt(self.ms_out - np.square(self.mg_out) + self.epsilon)
            self.inputs["MeanGrad"] = self.mean_grad
        else:
            self.moment_out = self.momentum * self.moment + \
                              self.learning_rate * self.grad / np.sqrt(self.ms_out + self.epsilon)

        self.param_out = self.param - self.moment_out

        self.outputs = {
            'ParamOut': np.array(self.param_out),
            "MomentOut": np.array(self.moment_out),
            "MeanSquareOut": np.array(self.ms_out),
            "MeanGradOut": self.mg_out
        }

    def set_npu(self):
        self.__class__.use_npu = True

    def set_centered(self):
        self.centered = False

    def set_size(self):
        self.size = (128, 320)

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-5)


class TestRmspropOpCentered(TestRMSProp):
    def set_centered(self):
        self.centered = False


class TestRmspropOpSize(TestRMSProp):
    def set_size(self):
        self.size = (102, 105)


class TestRMSPropNet(unittest.TestCase):
    def test_rmsprop(self):
        place = paddle.NPUPlace(0)
        main = fluid.Program()
        with fluid.program_guard(main):
            x = fluid.layers.data(name='x', shape=[13], dtype='float32')
            y = fluid.layers.data(name='y', shape=[1], dtype='float32')
            y_predict = fluid.layers.fc(input=x, size=1, act=None)
            cost = fluid.layers.square_error_cost(input=y_predict, label=y)
            avg_cost = fluid.layers.mean(cost)

            rms_optimizer = paddle.optimizer.RMSProp(learning_rate=0.1)
            rms_optimizer.minimize(avg_cost)

            fetch_list = [avg_cost]
            train_reader = paddle.batch(
                paddle.dataset.uci_housing.train(), batch_size=1)
            feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for data in train_reader():
                exe.run(main, feed=feeder.feed(data), fetch_list=fetch_list)


class TestNet(unittest.TestCase):
    def _test(self, run_npu=True):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        np.random.seed(SEED)

        a_np = np.random.random(size=(32, 32)).astype('float32')
        b_np = np.random.random(size=(32, 32)).astype('float32')
        label_np = np.random.randint(2, size=(32, 1)).astype('int64')

        with paddle.static.program_guard(main_prog, startup_prog):
            a = paddle.static.data(name="a", shape=[32, 32], dtype='float32')
            b = paddle.static.data(name="b", shape=[32, 32], dtype='float32')
            label = paddle.static.data(
                name="label", shape=[32, 1], dtype='int64')

            sum = paddle.add(a, b)
            z = paddle.pow(sum, 2.0)

            fc_1 = fluid.layers.fc(input=z, size=128)
            prediction = fluid.layers.fc(input=fc_1, size=2, act='softmax')

            cost = fluid.layers.cross_entropy(input=prediction, label=label)
            loss = fluid.layers.reduce_mean(cost)
            rmsprop = fluid.optimizer.RMSProp(learning_rate=0.01)
            rmsprop.minimize(loss)

        if run_npu:
            place = paddle.NPUPlace(0)
        else:
            place = paddle.CPUPlace()

        exe = paddle.static.Executor(place)
        exe.run(startup_prog)

        print("Start run on {}".format(place))
        for epoch in range(100):

            pred_res, loss_res = exe.run(
                main_prog,
                feed={"a": a_np,
                      "b": b_np,
                      "label": label_np},
                fetch_list=[prediction, loss])
            if epoch % 10 == 0:
                print("Epoch {} | Prediction[0]: {}, Loss: {}".format(
                    epoch, pred_res[0], loss_res))

        return pred_res, loss_res

    def test_npu(self):
        cpu_pred, cpu_loss = self._test(False)
        npu_pred, npu_loss = self._test(True)

        self.assertTrue(np.allclose(npu_pred, cpu_pred, rtol=1e-3))
        self.assertTrue(np.allclose(npu_loss, cpu_loss, rtol=1e-3))


class TestCenteredNet(unittest.TestCase):
    def _test(self, run_npu=True):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        np.random.seed(SEED)

        a_np = np.random.random(size=(32, 32)).astype('float32')
        b_np = np.random.random(size=(32, 32)).astype('float32')
        label_np = np.random.randint(2, size=(32, 1)).astype('int64')

        with paddle.static.program_guard(main_prog, startup_prog):
            a = paddle.static.data(name="a", shape=[32, 32], dtype='float32')
            b = paddle.static.data(name="b", shape=[32, 32], dtype='float32')
            label = paddle.static.data(
                name="label", shape=[32, 1], dtype='int64')

            sum = paddle.add(a, b)
            z = paddle.pow(sum, 2.0)

            fc_1 = fluid.layers.fc(input=z, size=128)
            prediction = fluid.layers.fc(input=fc_1, size=2, act='softmax')

            cost = fluid.layers.cross_entropy(input=prediction, label=label)
            loss = fluid.layers.reduce_mean(cost)
            rmsprop = fluid.optimizer.RMSProp(learning_rate=0.01, centered=True)
            rmsprop.minimize(loss)

        if run_npu:
            place = paddle.NPUPlace(0)
        else:
            place = paddle.CPUPlace()

        exe = paddle.static.Executor(place)
        exe.run(startup_prog)

        print("Start run on {}".format(place))
        for epoch in range(100):

            pred_res, loss_res = exe.run(
                main_prog,
                feed={"a": a_np,
                      "b": b_np,
                      "label": label_np},
                fetch_list=[prediction, loss])
            if epoch % 10 == 0:
                print("Epoch {} | Prediction[0]: {}, Loss: {}".format(
                    epoch, pred_res[0], loss_res))

        return pred_res, loss_res

    def test_npu(self):
        cpu_pred, cpu_loss = self._test(False)
        npu_pred, npu_loss = self._test(True)

        self.assertTrue(np.allclose(npu_pred, cpu_pred, rtol=1e-3))
        self.assertTrue(np.allclose(npu_loss, cpu_loss, rtol=1e-3))


if __name__ == "__main__":
    unittest.main()
