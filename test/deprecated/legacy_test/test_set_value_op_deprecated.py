#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

# Test set_value op in static graph mode

import unittest
from functools import reduce

import numpy as np

import paddle
from paddle.base.layer_helper import LayerHelper


class TestBackward(unittest.TestCase):
    def test_static(self):
        paddle.enable_static()
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()

        x_np = np.random.random(size=(4, 4)).astype('float32')
        y_np = np.random.random(size=(4, 4)).astype('float32')
        label_np = np.random.randint(2, size=(4, 1)).astype('int64')

        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data(name="x", shape=[4, 4], dtype='float32')
            y = paddle.static.data(name="y", shape=[4, 4], dtype='float32')
            x.stop_gradient = False
            y.stop_gradient = False

            label = paddle.static.data(
                name="label", shape=[4, 1], dtype='int64'
            )

            z = paddle.add(x, y)
            var = y[0, :]
            z = paddle.static.setitem(z, (0, slice(None)), var)

            prediction = paddle.static.nn.fc(x=z, size=2, activation='softmax')

            cost = paddle.nn.functional.cross_entropy(
                input=prediction, label=label
            )
            loss = paddle.mean(cost)
            sgd = paddle.optimizer.SGD(learning_rate=0.01)
            sgd.minimize(loss)

        exe = paddle.static.Executor(paddle.CPUPlace())
        exe.run(startup_program)

        var_grad, z_grad = exe.run(
            main_program,
            feed={"x": x_np, "y": y_np, "label": label_np},
            fetch_list=[var.name + "@GRAD", z.name + "@GRAD"],
        )

        self.assertTrue((var_grad == z_grad[0, :]).all())
        paddle.disable_static()


class TestGradientTruncated(unittest.TestCase):
    def test_static_graph(self):
        paddle.enable_static()

        to_string = lambda x, i: x + '_' + str(i)
        numel = lambda input_shape: reduce(lambda x, y: x * y, input_shape, 1)

        def op1(x):
            value = paddle.tensor.fill_constant([1], "float32", 1)
            # test stop_gradient
            value.stop_gradient = True
            x.stop_gradient = False
            start = paddle.tensor.fill_constant([1], "int32", 5, force_cpu=True)
            end = paddle.tensor.fill_constant([1], "int32", 0, force_cpu=True)
            step = paddle.tensor.fill_constant([1], "int32", -2, force_cpu=True)

            inputs = {
                'Input': x,
                'ValueTensor': value,
                'StartsTensorList': [
                    start,
                ],
                'EndsTensorList': [
                    end,
                ],
                'StepsTensorList': [
                    step,
                ],
            }

            helper = LayerHelper("set_value")
            y = helper.create_variable_for_type_inference(dtype=x.dtype)

            helper.append_op(
                type="set_value",
                inputs=inputs,
                outputs={'Out': y},
                attrs={'axes': [0]},
            )

            return y, value

        def op2(x):
            value = paddle.tensor.fill_constant([1, 3, 2], "float32", 1)
            # test stop_gradient
            value.stop_gradient = False
            x.stop_gradient = False
            attrs = {
                'axes': [0],
                'starts': [6],
                'ends': [0],
                'steps': [-4],
                'decrease_axes': [],
                'none_axes': [],
                'dtype': paddle.float32,
            }
            inputs = {'Input': x, 'ValueTensor': value}

            helper = LayerHelper("set_value")
            y = helper.create_variable_for_type_inference(dtype=x.dtype)

            helper.append_op(
                type="set_value", inputs=inputs, outputs={'Out': y}, attrs=attrs
            )

            return y, value

        def op3(x):
            value = paddle.tensor.fill_constant([1], "float32", 1)
            x.stop_gradient = True
            value.stop_gradient = False
            start = paddle.tensor.fill_constant([1], "int32", 0, force_cpu=True)
            end = paddle.tensor.fill_constant([1], "int32", 5, force_cpu=True)
            step = paddle.tensor.fill_constant([1], "int32", 3, force_cpu=True)

            inputs = {
                'Input': x,
                'ValueTensor': value,
                'StartsTensorList': [
                    start,
                ],
                'EndsTensorList': [
                    end,
                ],
                'StepsTensorList': [
                    step,
                ],
            }

            helper = LayerHelper("set_value")
            y = helper.create_variable_for_type_inference(dtype=x.dtype)

            helper.append_op(
                type="set_value",
                inputs=inputs,
                outputs={'Out': y},
                attrs={'axes': [0]},
            )

            return y, value

        def set_value(array, i, op):
            name_x = to_string('x', i)
            x = paddle.static.data(
                name=name_x, shape=array.shape, dtype='float32'
            )

            # set_value_op in __get/setitem__ is an inplace operation.
            # When `input.stop_gradient = True` and `value.stop_gradient = False`,
            # set_value_grad_op will not be run during backward.
            y, value = op(x)
            y2 = y + 1
            loss = paddle.sum(y2)
            sgd = paddle.optimizer.Adam()
            sgd.minimize(loss)
            place = (
                paddle.base.CPUPlace()
                if not paddle.base.core.is_compiled_with_cuda()
                else paddle.base.CUDAPlace(0)
            )

            prog = paddle.static.default_main_program()
            exe = paddle.static.Executor(place)
            exe.run(paddle.static.default_startup_program())
            fetch_list = []
            if not x.stop_gradient:
                fetch_list.append(x.grad_name)
            if not value.stop_gradient:
                fetch_list.append(value.grad_name)
            out = exe.run(prog, feed={x.name: array}, fetch_list=fetch_list)
            return out

        input_shape = [7, 6, 5, 4, 3, 2]

        array = np.arange(0, numel(input_shape), dtype="float32").reshape(
            input_shape
        )

        for i in range(len(input_shape)):
            program = paddle.static.Program()
            with paddle.static.program_guard(program):
                out1 = set_value(array, i, op1)
                self.assertTrue((out1[0][5:0:-2] == 0).all())

            if len(array.shape) > 2:
                program2 = paddle.static.Program()
                with paddle.static.program_guard(program2):
                    out2 = set_value(array, i, op2)
                    self.assertTrue((out2[0][6:0:-4] == 0).all())

            program3 = paddle.static.Program()
            with paddle.static.program_guard(program3):
                out3 = set_value(array, i, op3)
                self.assertTrue((numel(out1[0][0:5:3].shape) == out3[0]).all())

            array = array[0]
        paddle.disable_static()


class TestSetValueWithScalarInStatic(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        self.shape = (10, 2)
        self.exe = paddle.static.Executor()
        self.train_program = paddle.static.Program()
        self.startup_program = paddle.static.Program()

    def test_value_input_is_scalar(self):
        with paddle.static.program_guard(
            self.train_program, self.startup_program
        ):
            x = paddle.ones(self.shape)
            x.stop_gradient = False
            y = x * 1

            # mock test case x[0, 0] = 10 with no ValueTensor input
            inputs = {
                'Input': y,
            }
            attrs = {
                'axes': [0, 1],
                'starts': [0, 0],
                'ends': [1, 1],
                'steps': [1, 1],
                'values': [10],
                'shape': [1],
            }

            helper = LayerHelper("set_value")
            out = helper.create_variable_for_type_inference(dtype=y.dtype)

            helper.append_op(
                type="set_value",
                inputs=inputs,
                outputs={'Out': out},
                attrs=attrs,
            )

            np_data = np.ones(self.shape).astype('float32')

            paddle.static.append_backward(out.sum())
            res = self.exe.run(
                self.train_program, fetch_list=[out, x.grad_name]
            )

            np_data[0, 0] = 10
            expected_x_grad = np.ones(self.shape)
            expected_x_grad[0, 0] = 0

        np.testing.assert_array_equal(res[0], np_data)
        np.testing.assert_array_equal(res[1], expected_x_grad)


if __name__ == '__main__':
    unittest.main()
