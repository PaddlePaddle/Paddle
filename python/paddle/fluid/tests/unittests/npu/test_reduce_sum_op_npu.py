#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

<<<<<<< HEAD
=======
from __future__ import print_function

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import numpy as np
import unittest
import sys

sys.path.append("..")
from op_test import OpTest
import paddle
import paddle.fluid as fluid

paddle.enable_static()
SEED = 2021


class TestReduceSum(OpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        np.random.seed(SEED)
        self.set_npu()
        self.init_dtype()
        self.place = paddle.NPUPlace(0)
        self.init_op_type()
        self.initTestCase()

        self.use_mkldnn = False
        self.attrs = {
            'dim': self.axis,
            'keep_dim': self.keep_dim,
<<<<<<< HEAD
            'reduce_all': self.reduce_all,
=======
            'reduce_all': self.reduce_all
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }
        self.inputs = {'X': np.random.random(self.shape).astype(self.dtype)}
        if self.attrs['reduce_all']:
            self.outputs = {'Out': self.inputs['X'].sum()}
        else:
            self.outputs = {
<<<<<<< HEAD
                'Out': self.inputs['X'].sum(
                    axis=self.axis, keepdims=self.attrs['keep_dim']
                )
=======
                'Out':
                self.inputs['X'].sum(axis=self.axis,
                                     keepdims=self.attrs['keep_dim'])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            }

    def set_npu(self):
        self.__class__.use_npu = True

    def init_dtype(self):
        self.dtype = np.float32

    def init_op_type(self):
        self.op_type = "reduce_sum"
        self.use_mkldnn = False
        self.keep_dim = False
        self.reduce_all = False

    def initTestCase(self):
        self.shape = (5, 6)
<<<<<<< HEAD
        self.axis = 0
=======
        self.axis = (0)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_check_output(self):
        self.check_output_with_place(self.place)

    # TODO(ascendrc): Add grad test
    # def test_check_grad(self):
    #     if self.dtype == np.float16:
    #         return
    #     self.check_grad(['X'], 'Out')
    #


class TestReduceSum2(OpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_dtype(self):
        self.dtype = np.int32


class TestReduceSumNet(unittest.TestCase):
<<<<<<< HEAD
    def set_reduce_sum_function(self, x):
        # keep_dim = False
        return paddle.sum(x, axis=-1)
=======

    def set_reduce_sum_function(self, x):
        # keep_dim = False
        return paddle.fluid.layers.reduce_sum(x, dim=-1)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def _test(self, run_npu=True):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        np.random.seed(SEED)

        a_np = np.random.random(size=(2, 3, 4)).astype('float32')
        b_np = np.random.random(size=(2, 3, 4)).astype('float32')
        label_np = np.random.randint(2, size=(2, 1)).astype('int64')

        with paddle.static.program_guard(main_prog, startup_prog):
            a = paddle.static.data(name="a", shape=[2, 3, 4], dtype='float32')
            b = paddle.static.data(name="b", shape=[2, 3, 4], dtype='float32')
<<<<<<< HEAD
            label = paddle.static.data(
                name="label", shape=[2, 1], dtype='int64'
            )

            a_1 = paddle.static.nn.fc(x=a, size=4, num_flatten_dims=2, activation=None)
            b_1 = paddle.static.nn.fc(x=b, size=4, num_flatten_dims=2, activation=None)
            z = paddle.add(a_1, b_1)
            z_1 = self.set_reduce_sum_function(z)

            prediction = paddle.static.nn.fc(x=z_1, size=2, activation='softmax')

            cost = paddle.nn.functional.cross_entropy(input=prediction, label=label, reduction='none', use_softmax=False)
            loss = paddle.mean(cost)
=======
            label = paddle.static.data(name="label",
                                       shape=[2, 1],
                                       dtype='int64')

            a_1 = fluid.layers.fc(input=a, size=4, num_flatten_dims=2, act=None)
            b_1 = fluid.layers.fc(input=b, size=4, num_flatten_dims=2, act=None)
            z = paddle.add(a_1, b_1)
            z_1 = self.set_reduce_sum_function(z)

            prediction = fluid.layers.fc(input=z_1, size=2, act='softmax')

            cost = fluid.layers.cross_entropy(input=prediction, label=label)
            loss = fluid.layers.reduce_mean(cost)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            sgd = fluid.optimizer.SGD(learning_rate=0.01)
            sgd.minimize(loss)

        if run_npu:
            place = paddle.NPUPlace(0)
        else:
            place = paddle.CPUPlace()

        exe = paddle.static.Executor(place)
        exe.run(startup_prog)

        print("Start run on {}".format(place))
        for epoch in range(100):

<<<<<<< HEAD
            pred_res, loss_res = exe.run(
                main_prog,
                feed={"a": a_np, "b": b_np, "label": label_np},
                fetch_list=[prediction, loss],
            )
            if epoch % 10 == 0:
                print(
                    "Epoch {} | Prediction[0]: {}, Loss: {}".format(
                        epoch, pred_res[0], loss_res
                    )
                )
=======
            pred_res, loss_res = exe.run(main_prog,
                                         feed={
                                             "a": a_np,
                                             "b": b_np,
                                             "label": label_np
                                         },
                                         fetch_list=[prediction, loss])
            if epoch % 10 == 0:
                print("Epoch {} | Prediction[0]: {}, Loss: {}".format(
                    epoch, pred_res[0], loss_res))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        return pred_res, loss_res

    def test_npu(self):
        cpu_pred, cpu_loss = self._test(False)
        npu_pred, npu_loss = self._test(True)

        np.testing.assert_allclose(npu_pred, cpu_pred)
        np.testing.assert_allclose(npu_loss, cpu_loss)


class TestReduceSumNet2(TestReduceSumNet):
<<<<<<< HEAD
    def set_reduce_sum_function(self, x):
        # keep_dim = True
        return paddle.sum(x, axis=-1, keepdim=True)


class TestReduceSumNet3(TestReduceSumNet):
=======

    def set_reduce_sum_function(self, x):
        # keep_dim = True
        return paddle.fluid.layers.reduce_sum(x, dim=-1, keep_dim=True)


class TestReduceSumNet3(TestReduceSumNet):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def _test(self, run_npu=True):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        np.random.seed(SEED)

        a_np = np.random.random(size=(2, 3, 4)).astype('float32')
        b_np = np.random.random(size=(2, 3, 4)).astype('float32')

        with paddle.static.program_guard(main_prog, startup_prog):
            a = paddle.static.data(name="a", shape=[2, 3, 4], dtype='float32')
            b = paddle.static.data(name="b", shape=[2, 3, 4], dtype='float32')

            z = paddle.add(a, b)
<<<<<<< HEAD
            loss = paddle.sum(z)
=======
            loss = fluid.layers.reduce_sum(z)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            sgd = fluid.optimizer.SGD(learning_rate=0.01)
            sgd.minimize(loss)

        if run_npu:
            place = paddle.NPUPlace(0)
        else:
            place = paddle.CPUPlace()

        exe = paddle.static.Executor(place)
        exe.run(startup_prog)

        print("Start run on {}".format(place))
        for epoch in range(100):

<<<<<<< HEAD
            loss_res = exe.run(
                main_prog, feed={"a": a_np, "b": b_np}, fetch_list=[loss]
            )
=======
            loss_res = exe.run(main_prog,
                               feed={
                                   "a": a_np,
                                   "b": b_np
                               },
                               fetch_list=[loss])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            if epoch % 10 == 0:
                print("Epoch {} | Loss: {}".format(epoch, loss_res))

        return loss_res, loss_res


if __name__ == '__main__':
    unittest.main()
