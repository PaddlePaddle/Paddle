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


def ComputeGrad(x, y, out, axis):
    grad = 1 / out.size
    shape_x = x.shape
    shape_y = y.shape
    shape_out = out.shape
    reduce_axes_x = []
    reduce_axes_y = []

    if shape_x != shape_out:
        if len(shape_x.shape) < len(shape_out.shape):
            src_axis = axis
        else:
            src_axis = 0

        for ax in range(len(shape_out)):
            if (ax < src_axis or ax >= src_axis + len(shape_x)) or (
                    shape_out[ax] > 1 and shape_x[ax - src_axis] == 1):
                reduce_axes_x.append(ax)

    if shape_y != shape_out:
        if len(shape_y) < len(shape_out):
            src_axis = axis
        else:
            src_axis = 0

        for ax in range(len(shape_out)):
            if (ax < src_axis or ax >= src_axis + len(shape_y)) or (
                    shape_out[ax] > 1 and shape_y[ax - src_axis] == 1):
                reduce_axes_y.append(ax)

    if len(reduce_axes_x) > 0:
        for i in reduce_axes_x:
            x = np.expand_dims(x, axis=i)

    if len(reduce_axes_y) > 0:
        for i in reduce_axes_y:
            y = np.expand_dims(y, axis=i)

    mask = np.sign(np.subtract(x, y))
    dx = np.maximum(mask, 0) * grad
    dy = np.abs(np.minimum(mask, 0) * grad)

    if len(reduce_axes_x) > 0:
        for i, element in enumerate(reduce_axes_x):
            dx = np.add.reduce(dx, element - i)

    if len(reduce_axes_y) > 0:
        for i, element in enumerate(reduce_axes_y):
            dy = np.add.reduce(dy, element - i)

    return dx, dy


class TestElementwiseMaxOp(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "elementwise_max"
        self.place = paddle.NPUPlace(0)

        self.init_dtype()
        self.init_input_output()
        self.init_axis()

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

    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        sgn = np.random.choice([-1, 1], [13, 17]).astype(self.dtype)
        self.y = self.x + sgn * np.random.uniform(0.1, 1,
                                                  [13, 17]).astype(self.dtype)
        self.out = np.maximum(self.x, self.y)

    def init_axis(self):
        self.axis = -1

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        self.check_grad_with_place(self.place, ['X', 'Y'], 'Out')

    def test_check_grad_ingore_x(self):
        self.check_grad_with_place(
            self.place, ['Y'], 'Out', no_grad_set=set("X"))

    def test_check_grad_ingore_y(self):
        self.check_grad_with_place(
            self.place, ['X'], 'Out', no_grad_set=set("Y"))


class TestElementwiseMaxOp_int32(TestElementwiseMaxOp):
    def init_dtype(self):
        self.dtype = np.int32

    # CTest does not support check grad for int32.
    def test_check_grad_normal(self):
        pass

    def test_check_grad_ingore_x(self):
        pass

    def test_check_grad_ingore_y(self):
        pass


class TestElementwiseMaxOp_scalar(TestElementwiseMaxOp):
    def init_input_output(self):
        self.x = np.random.random_integers(-5, 5, [2, 3, 20]).astype(self.dtype)
        self.y = np.array([0.5]).astype(self.dtype)
        self.out = np.maximum(self.x, self.y)


class TestElementwiseMaxOp_vector(TestElementwiseMaxOp):
    def init_input_output(self):
        self.x = np.random.random((100, )).astype(self.dtype)
        sgn = np.random.choice([-1, 1], (100, )).astype(self.dtype)
        self.y = self.x + sgn * np.random.uniform(0.1, 1,
                                                  (100, )).astype(self.dtype)
        self.out = np.maximum(self.x, self.y)


class TestElementwiseMaxOp_broadcast_0(TestElementwiseMaxOp):
    def init_input_output(self):
        self.x = np.random.uniform(0.5, 1, (100, 5, 2)).astype(self.dtype)
        sgn = np.random.choice([-1, 1], (100, )).astype(self.dtype)
        self.y = self.x[:, 0, 0] + sgn * \
            np.random.uniform(1, 2, (100, )).astype(self.dtype)
        self.out = np.maximum(self.x, self.y.reshape(100, 1, 1))

    def init_axis(self):
        self.axis = 0


class TestElementwiseMaxOp_broadcast_1(TestElementwiseMaxOp):
    def init_input_output(self):
        self.x = np.random.uniform(0.5, 1, (2, 100, 3)).astype(self.dtype)
        sgn = np.random.choice([-1, 1], (100, )).astype(self.dtype)
        self.y = self.x[0, :, 0] + sgn * \
            np.random.uniform(1, 2, (100, )).astype(self.dtype)
        self.out = np.maximum(self.x, self.y.reshape(1, 100, 1))

    def init_axis(self):
        self.axis = 1

    def test_check_grad_ingore_x(self):
        _, dy = ComputeGrad(self.x, self.y, self.out, self.axis)
        self.check_grad_with_place(
            self.place, ['Y'],
            'Out',
            no_grad_set=set("X"),
            user_defined_grads=[dy])

    def test_check_grad_ingore_y(self):
        dx, _ = ComputeGrad(self.x, self.y, self.out, self.axis)
        self.check_grad_with_place(
            self.place, ['X'],
            'Out',
            no_grad_set=set("Y"),
            user_defined_grads=[dx])


class TestElementwiseMaxOp_broadcast_2(TestElementwiseMaxOp):
    def init_input_output(self):
        self.x = np.random.uniform(0.5, 1, (2, 3, 100)).astype(self.dtype)
        sgn = np.random.choice([-1, 1], (100, )).astype(self.dtype)
        self.y = self.x[0, 0, :] + sgn * \
            np.random.uniform(1, 2, (100, )).astype(self.dtype)
        self.out = np.maximum(self.x, self.y.reshape(1, 1, 100))

    def test_check_grad_normal(self):
        dx, dy = ComputeGrad(self.x, self.y, self.out, self.axis)
        self.check_grad_with_place(
            self.place, ['X', 'Y'], 'Out', user_defined_grads=[dx, dy])

    def test_check_grad_ingore_x(self):
        _, dy = ComputeGrad(self.x, self.y, self.out, self.axis)
        self.check_grad_with_place(
            self.place, ['Y'],
            'Out',
            no_grad_set=set("X"),
            user_defined_grads=[dy])

    def test_check_grad_ingore_y(self):
        dx, _ = ComputeGrad(self.x, self.y, self.out, self.axis)
        self.check_grad_with_place(
            self.place, ['X'],
            'Out',
            no_grad_set=set("Y"),
            user_defined_grads=[dx])


class TestElementwiseMaxOp_broadcast_3(TestElementwiseMaxOp):
    def init_input_output(self):
        self.x = np.random.uniform(0.5, 1, (2, 50, 2, 1)).astype(self.dtype)
        sgn = np.random.choice([-1, 1], (50, 2)).astype(self.dtype)
        self.y = self.x[0, :, :, 0] + sgn * \
            np.random.uniform(1, 2, (50, 2)).astype(self.dtype)
        self.out = np.maximum(self.x, self.y.reshape(1, 50, 2, 1))

    def init_axis(self):
        self.axis = 1


class TestElementwiseMaxOp_broadcast_4(TestElementwiseMaxOp):
    def init_input_output(self):
        self.x = np.random.uniform(0.5, 1, (2, 3, 4, 5)).astype(self.dtype)
        sgn = np.random.choice([-1, 1], (2, 3, 1, 5)).astype(self.dtype)
        self.y = self.x + sgn * \
            np.random.uniform(1, 2, (2, 3, 1, 5)).astype(self.dtype)
        self.out = np.maximum(self.x, self.y)


class TestElementwiseMaxOp_broadcast_5(TestElementwiseMaxOp):
    def init_input_output(self):
        self.x = np.random.uniform(0.5, 1, (2, 3, 4, 5)).astype(self.dtype)
        sgn = np.random.choice([-1, 1], (2, 3, 1, 1)).astype(self.dtype)
        self.y = self.x + sgn * \
            np.random.uniform(1, 2, (2, 3, 1, 1)).astype(self.dtype)
        self.out = np.maximum(self.x, self.y)


class TestElementwiseMaxNet(unittest.TestCase):
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

            c = paddle.maximum(a, b)

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

        self.assertTrue(np.allclose(npu_pred, cpu_pred))
        self.assertTrue(np.allclose(npu_loss, cpu_loss))


if __name__ == '__main__':
    unittest.main()
