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
from op_test import OpTest
import paddle
import paddle.fluid as fluid

paddle.enable_static()

SEED = 2021


class TestElementwiseSubOp(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "elementwise_sub"
        self.place = paddle.NPUPlace(0)
        self.init_dtype()
        self.init_input_output()
        self.init_kernel_type()
        self.init_axis()

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(self.x),
            'Y': OpTest.np_dtype_to_fluid_dtype(self.y),
        }
        self.attrs = {'axis': self.axis, 'use_mkldnn': self.use_mkldnn}
        self.outputs = {'Out': self.out}

    def set_npu(self):
        self.__class__.use_npu = True

    def init_kernel_type(self):
        self.use_mkldnn = False

    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.out = np.subtract(self.x, self.y)

    def init_dtype(self):
        self.dtype = np.float32

    def init_axis(self):
        self.axis = 0

    def test_check_output(self):
        self.check_output_with_place(self.place)

    # TODO(ascendrc): For grad tests, OpTest raises FatalError:Segmentation fault
    #  when call op.run, which may be caused by system environment exception
    #  and the exact cause has not be located.
    # def test_check_grad_normal(self):
    #     self.check_grad_with_place(
    #         self.place, ['X', 'Y'],
    #         'Out',
    #         max_relative_error=0.006,
    #         )
    #
    # def test_check_grad_ingore_x(self):
    #     self.check_grad_with_place(
    #         self.place, ['Y'],
    #         'Out',
    #         no_grad_set=set("X"),
    #         max_relative_error=0.006,
    #         )
    #
    # def test_check_grad_ingore_y(self):
    #     self.check_grad_with_place(
    #         self.place, ['X'],
    #         'Out',
    #         no_grad_set=set("Y"),
    #         max_relative_error=0.006,)


class TestElementwiseSubOpInt32(TestElementwiseSubOp):
    def init_dtype(self):
        self.dtype = np.int32


class TestElementwiseSubOpInt64(TestElementwiseSubOp):
    def init_dtype(self):
        self.dtype = np.int64


class TestSubtractAPI(unittest.TestCase):
    def test_name(self):
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name="x", shape=[2, 3], dtype="float32")
            y = paddle.static.data(name='y', shape=[2, 3], dtype='float32')

            y_1 = paddle.subtract(x, y, name='add_res')
            self.assertEqual(('add_res' in y_1.name), True)

    def test_static(self):
        with paddle.static.program_guard(paddle.static.Program()):

            x_np = np.array([2, 3, 4]).astype('float32')
            y_np = np.array([1, 5, 2]).astype('float32')

            x = paddle.static.data(name="x", shape=[3], dtype='float32')
            y = paddle.static.data(name="y", shape=[3], dtype='float32')

            x_reshape = paddle.reshape(x, [3, 1])
            y_reshape = paddle.reshape(y, [3, 1])
            z = paddle.subtract(x_reshape, y_reshape)
            z = paddle.reshape(z, shape=[3])

            place = paddle.NPUPlace(0)
            exe = paddle.static.Executor(place)
            x_value, y_value, z_value = exe.run(
                feed={"x": x_np, "y": y_np}, fetch_list=[x, y, z]
            )

            z_expected = np.array([1.0, -2.0, 2.0])
            self.assertEqual(
                (x_value == x_np).all(),
                True,
                msg="x_value = {}, but expected {}".format(x_value, x_np),
            )
            self.assertEqual(
                (y_value == y_np).all(),
                True,
                msg="y_value = {}, but expected {}".format(y_value, y_np),
            )
            self.assertEqual(
                (z_value == z_expected).all(),
                True,
                msg="z_value = {}, but expected {}".format(z_value, z_expected),
            )


class TestSubtractError(unittest.TestCase):
    def test_errors(self):
        with paddle.static.program_guard(paddle.static.Program()):
            # the input of elementwise_add must be Variable.
            x1 = fluid.create_lod_tensor(
                np.array([-1, 3, 5, 5]), [[1, 1, 1, 1]], fluid.NPUPlace(0)
            )
            y1 = fluid.create_lod_tensor(
                np.array([-1, 3, 5, 5]), [[1, 1, 1, 1]], fluid.NPUPlace(0)
            )
            self.assertRaises(TypeError, paddle.subtract, x1, y1)

            # the input dtype must be float16 or float32 or float64 or int32 or int64
            x2 = paddle.static.data(
                name='x2', shape=[3, 4, 5, 6], dtype="uint8"
            )
            y2 = paddle.static.data(
                name='y2', shape=[3, 4, 5, 6], dtype="uint8"
            )
            self.assertRaises(TypeError, paddle.subtract, x2, y2)


class TestSubtractNet(unittest.TestCase):
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
                name="label", shape=[32, 1], dtype='int64'
            )

            sum = paddle.add(a, b)
            c = paddle.assign(b)
            z = paddle.subtract(sum, c)

            fc_1 = fluid.layers.fc(input=z, size=128)
            prediction = fluid.layers.fc(input=fc_1, size=2, act='softmax')

            cost = paddle.nn.functional.cross_entropy(input=prediction, label=label, reduction='none', use_softmax=False)
            loss = paddle.mean(cost)
            sgd = fluid.optimizer.SGD(learning_rate=0.01)
            sgd.minimize(loss)

        if run_npu:
            place = paddle.NPUPlace(0)
        else:
            place = paddle.CPUPlace()

        exe = paddle.static.Executor(place)
        exe.run(startup_prog)

        for epoch in range(100):

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

        return pred_res, loss_res

    def test_npu(self):
        npu_pred, npu_loss = self._test(True)
        cpu_pred, cpu_loos = self._test(False)

        np.testing.assert_allclose(npu_pred, cpu_pred, rtol=1e-6)
        np.testing.assert_allclose(npu_loss, cpu_loos, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
