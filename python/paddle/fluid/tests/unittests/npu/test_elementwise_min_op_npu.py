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

import numpy as np
import unittest
import sys

sys.path.append("..")
from op_test import OpTest, skip_check_grad_ci
import paddle
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard
import paddle.fluid.core as core

paddle.enable_static()
SEED = 2021


class TestElementwiseMinOp(OpTest):

    def setUp(self):
        self.set_npu()
        self.op_type = "elementwise_min"
        self.place = paddle.NPUPlace(0)
        self.init_dtype()
        self.init_input_output()
        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(self.x),
            'Y': OpTest.np_dtype_to_fluid_dtype(self.y)
        }
        self.outputs = {'Out': self.out}
        self.attrs = {'axis': self.axis}

    def set_npu(self):
        self.__class__.use_npu = True

    def init_input_output(self):
        # If x and y have the same value, the min() is not differentiable.
        # So we generate test data by the following method
        # to avoid them being too close to each other.
        self.x = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.sgn = np.random.choice([-1, 1], [13, 17]).astype(self.dtype)
        self.y = self.x + self.sgn * np.random.uniform(0.1, 1, [13, 17]).astype(
            self.dtype)
        self.out = np.minimum(self.x, self.y)
        self.axis = -1

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        if self.dtype == np.float16:
            self.check_grad_with_place(self.place, ['X', 'Y'],
                                       'Out',
                                       max_relative_error=0.5)
        else:
            self.check_grad_with_place(
                self.place,
                ['X', 'Y'],
                'Out',
            )

    def test_check_grad_ingore_x(self):
        if self.dtype == np.float16:
            self.check_grad_with_place(self.place, ['Y'],
                                       'Out',
                                       no_grad_set=set("X"),
                                       max_relative_error=0.9)
        else:
            self.check_grad_with_place(
                self.place,
                ['Y'],
                'Out',
                no_grad_set=set("X"),
            )

    def test_check_grad_ingore_y(self):
        if self.dtype == np.float16:
            self.check_grad_with_place(self.place, ['X'],
                                       'Out',
                                       no_grad_set=set("Y"),
                                       max_relative_error=0.1)
        else:
            self.check_grad_with_place(
                self.place,
                ['X'],
                'Out',
                no_grad_set=set("Y"),
            )


class TestElementwiseMinOpFp16(TestElementwiseMinOp):

    def init_dtype(self):
        self.dtype = np.float16


class TestElementwiseMinOp_Vector(TestElementwiseMinOp):

    def init_input_output(self):
        self.x = np.random.uniform(1, 2, (100, )).astype(self.dtype)
        self.sgn = np.random.choice([-1, 1], (100, )).astype(self.dtype)
        self.y = self.x + self.sgn * np.random.uniform(0.1, 1, (100, )).astype(
            self.dtype)
        self.out = np.minimum(self.x, self.y)
        self.axis = -1


class TestElementwiseMinOpFp16_Vector(TestElementwiseMinOp_Vector):

    def init_dtype(self):
        self.dtype = np.float16


@skip_check_grad_ci(
    reason="[skip shape check] Use y_shape(1) to test broadcast.")
class TestElementwiseMinOp_scalar(TestElementwiseMinOp):

    def init_input_output(self):
        self.x = np.random.random_integers(-5, 5, [10, 3, 4]).astype(self.dtype)
        self.y = np.array([0.5]).astype(self.dtype)
        self.out = np.minimum(self.x, self.y)
        self.axis = -1


@skip_check_grad_ci(
    reason="[skip shape check] Use y_shape(1) to test broadcast.")
class TestElementwiseMinOpFp16_scalar(TestElementwiseMinOp_scalar):

    def init_dtype(self):
        self.dtype = np.float16


class TestElementwiseMinOp_broadcast(TestElementwiseMinOp):

    def init_input_output(self):
        self.x = np.random.uniform(0.5, 1, (2, 3, 100)).astype(self.dtype)
        self.sgn = np.random.choice([-1, 1], (100, )).astype(self.dtype)
        self.y = self.x[0, 0, :] + self.sgn * \
            np.random.uniform(1, 2, (100, )).astype(self.dtype)
        self.out = np.minimum(self.x, self.y.reshape(1, 1, 100))
        self.axis = -1


class TestElementwiseMinOpFp16_broadcast(TestElementwiseMinOp_broadcast):

    def init_dtype(self):
        self.dtype = np.float16


class TestElementwiseMinOpNet(unittest.TestCase):

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
            label = paddle.static.data(name="label",
                                       shape=[32, 1],
                                       dtype='int64')

            c = paddle.minimum(a, b)

            fc_1 = fluid.layers.fc(input=c, size=128)
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

        np.testing.assert_allclose(npu_pred, cpu_pred, rtol=1e-6)
        np.testing.assert_allclose(npu_loss, cpu_loss, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
