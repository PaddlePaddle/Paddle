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

from __future__ import print_function

import numpy as np
import unittest
import sys
sys.path.append("..")
from op_test import OpTest
import paddle
import paddle.fluid as fluid

paddle.enable_static()
SEED = 2021


class TestElementwiseMul(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "elementwise_mul"
        self.place = paddle.NPUPlace(0)
        self.init_axis()
        self.init_dtype()
        self.init_input_output()

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(self.x),
            'Y': OpTest.np_dtype_to_fluid_dtype(self.y)
        }
        self.attrs = {'axis': self.axis}
        self.outputs = {'Out': self.out}

    def set_npu(self):
        self.__class__.use_npu = True

    def init_dtype(self):
        self.dtype = np.float32

    def init_axis(self):
        self.axis = -1

    def init_input_output(self):
        np.random.seed(SEED)
        self.x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        self.y = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        self.out = np.multiply(self.x, self.y)

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ['X', 'Y'], 'Out')

    def test_check_grad_ignore_x(self):
        self.check_grad_with_place(
            self.place, ['Y'], 'Out', no_grad_set=set("X"))

    def test_check_grad_ignore_y(self):
        self.check_grad_with_place(
            self.place, ['X'], 'Out', no_grad_set=set("Y"))


class TestElementwiseMulFp16(TestElementwiseMul):
    def init_dtype(self):
        self.dtype = np.float16

    def init_input_output(self):
        np.random.seed(SEED)
        self.x = np.random.uniform(1, 2, [100, 4]).astype(self.dtype)
        self.y = np.random.uniform(1, 2, [100, 4]).astype(self.dtype)
        self.out = np.multiply(self.x, self.y)


class TestElementwiseMul_broadcast_0(TestElementwiseMul):
    def init_input_output(self):
        self.x = np.random.rand(100, 2, 3).astype(self.dtype)
        self.y = np.random.rand(100).astype(self.dtype)
        self.out = self.x * self.y.reshape(100, 1, 1)

    def init_axis(self):
        self.axis = 0


class TestElementwiseMul_broadcast_1(TestElementwiseMul):
    def init_input_output(self):
        self.x = np.random.rand(2, 100, 3).astype(self.dtype)
        self.y = np.random.rand(100).astype(self.dtype)
        self.out = self.x * self.y.reshape(1, 100, 1)

    def init_axis(self):
        self.axis = 1


class TestElementwiseMul_broadcast_2(TestElementwiseMul):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 100).astype(self.dtype)
        self.y = np.random.rand(100).astype(self.dtype)
        self.out = self.x * self.y.reshape(1, 1, 100)


class TestElementwiseMul_broadcast_3(TestElementwiseMul):
    def init_input_output(self):
        self.x = np.random.rand(2, 10, 12, 3).astype(self.dtype)
        self.y = np.random.rand(10, 12).astype(self.dtype)
        self.out = self.x * self.y.reshape(1, 10, 12, 1)

    def init_axis(self):
        self.axis = 1


class TestElementwiseMulOp_broadcast_4(TestElementwiseMul):
    def init_input_output(self):
        self.x = np.random.rand(10, 2, 11).astype(self.dtype)
        self.y = np.random.rand(10, 1, 11).astype(self.dtype)
        self.out = self.x * self.y


class TestElementwiseMulOp_broadcast_5(TestElementwiseMul):
    def init_input_output(self):
        self.x = np.random.rand(10, 4, 2, 3).astype(self.dtype)
        self.y = np.random.rand(10, 4, 1, 3).astype(self.dtype)
        self.out = self.x * self.y


class TestElementwiseMulNet(unittest.TestCase):
    def _test(self, run_npu=True):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        np.random.seed(SEED)

        a_np = np.random.random(size=(32, 32)).astype('float32')
        b_np = np.random.random(size=(32, 32)).astype('float32')
        c_np = np.random.random(size=(32, 32)).astype('float32')
        d_np = np.random.random(size=(32, 32)).astype('float32')
        label_np = np.random.randint(2, size=(32, 1)).astype('int64')

        with paddle.static.program_guard(main_prog, startup_prog):
            a = paddle.static.data(name="a", shape=[32, 32], dtype='float32')
            b = paddle.static.data(name="b", shape=[32, 32], dtype='float32')
            c = paddle.static.data(name="c", shape=[32, 32], dtype='float32')
            d = paddle.static.data(name="d", shape=[32, 32], dtype='float32')
            label = paddle.static.data(
                name="label", shape=[32, 1], dtype='int64')

            e = paddle.multiply(a, b)
            f = paddle.multiply(c, d)
            f.stop_gradient = True
            g = paddle.multiply(e, f)

            fc_1 = fluid.layers.fc(input=g, size=128)
            prediction = fluid.layers.fc(input=fc_1, size=2, act='softmax')

            cost = fluid.layers.cross_entropy(input=prediction, label=label)
            loss = fluid.layers.reduce_mean(cost)
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

            pred_res, loss_res = exe.run(main_prog,
                                         feed={
                                             "a": a_np,
                                             "b": b_np,
                                             "c": c_np,
                                             "d": d_np,
                                             "label": label_np
                                         },
                                         fetch_list=[prediction, loss])
            if epoch % 10 == 0:
                print("Epoch {} | Prediction[0]: {}, Loss: {}".format(
                    epoch, pred_res[0], loss_res))

        return pred_res, loss_res

    def test_npu(self):
        cpu_pred, cpu_loss = self._test(False)
        npu_pred, npu_loss = self._test(True)

        self.assertTrue(np.allclose(npu_pred, cpu_pred))
        self.assertTrue(np.allclose(npu_loss, cpu_loss))


if __name__ == '__main__':
    unittest.main()
