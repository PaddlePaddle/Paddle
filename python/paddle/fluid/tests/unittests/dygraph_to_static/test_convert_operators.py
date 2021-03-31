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

import numpy as np
import paddle
import unittest
from paddle.jit.dy2static.convert_operators import eval_if_exist_else_none


class CallNotExist(paddle.nn.Layer):
    def __call__(self):
        # call a non-exist API to trigger exception
        return paddle.nn.not_exist_api


class ForwardNotExist(paddle.nn.Layer):
    def forward(self):
        return 0


net = ForwardNotExist()
setattr(net, "forward", "A string so that convert forward will fail")


class TestConvertCall(unittest.TestCase):
    def test_class_exception(self):
        @paddle.jit.to_static
        def call_not_exist():
            net = CallNotExist()
            return net()

        with self.assertRaises(AttributeError):
            call_not_exist()

        @paddle.jit.to_static
        def forward_not_exist():
            return net()

        with self.assertRaises(TypeError):
            forward_not_exist()


class TestConvertShapeCompare(unittest.TestCase):
    def test_non_variable(self):
        self.assertEqual(
            paddle.jit.dy2static.convert_shape_compare(1, "<", 2), True)
        self.assertEqual(
            paddle.jit.dy2static.convert_shape_compare(1, "<", 2, "<=", 3),
            True)
        self.assertEqual(
            paddle.jit.dy2static.convert_shape_compare(1, ">", 2, "<=", 3),
            False)

        def error_func():
            """
            Function used to test that comparison doesn't run after first False
            """
            raise ValueError("Used for test")

        self.assertEqual(
            paddle.jit.dy2static.convert_shape_compare(
                1, ">", 2, "<=", lambda: error_func()), False)

        self.assertEqual(
            paddle.jit.dy2static.convert_shape_compare(1, "<", 2, "in",
                                                       [1, 2, 3]), True)
        self.assertEqual(
            paddle.jit.dy2static.convert_shape_compare(1, "<", 2, "not in",
                                                       [1, 2, 3]), False)
        self.assertEqual(
            paddle.jit.dy2static.convert_shape_compare(1, "<", 2, "is", 3),
            False)
        self.assertEqual(
            paddle.jit.dy2static.convert_shape_compare(1, "<", 2, "is not",
                                                       [1, 2, 3]), True)

        self.assertEqual(
            paddle.jit.dy2static.convert_shape_compare([1, 2], "==", [1, 2],
                                                       "!=", [1, 2, 3]), True)
        self.assertEqual(
            paddle.jit.dy2static.convert_shape_compare([1, 2], "!=", [1, 2, 3],
                                                       "==", [1, 2]), False)

    def test_variable(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program(),
                                         paddle.static.Program()):
            x = paddle.static.data(name='x', shape=[3, 2], dtype='float32')
            y = paddle.static.data(name='y', shape=[3, 2], dtype='float32')
            self.assertEqual(
                paddle.jit.dy2static.convert_shape_compare(x, "is", x, "is not",
                                                           y), True)
            self.assertEqual(
                paddle.jit.dy2static.convert_shape_compare(x, "is not", x,
                                                           "is not", y), False)
            self.assertEqual(
                paddle.jit.dy2static.convert_shape_compare(x, "is", x, "is", y),
                False)

            eq_out = paddle.jit.dy2static.convert_shape_compare(x, "==", y)
            not_eq_out = paddle.jit.dy2static.convert_shape_compare(x, "!=", y)
            long_eq_out = paddle.jit.dy2static.convert_shape_compare(x, "==", x,
                                                                     "!=", y)

            place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda(
            ) else paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            x_y_eq_out = exe.run(feed={
                "x": np.ones([3, 2]).astype(np.float32),
                "y": np.ones([3, 2]).astype(np.float32)
            },
                                 fetch_list=[eq_out, not_eq_out, long_eq_out])
            np.testing.assert_array_equal(
                np.array(x_y_eq_out), np.array([[True], [False], [False]]))

            set_a_zero = np.ones([3, 2]).astype(np.float32)
            set_a_zero[0][0] = 0.0
            x_y_not_eq_out = exe.run(
                feed={
                    "x": np.ones([3, 2]).astype(np.float32),
                    "y": set_a_zero
                },
                fetch_list=[eq_out, not_eq_out, long_eq_out])
            np.testing.assert_array_equal(
                np.array(x_y_not_eq_out), np.array([[False], [True], [True]]))
        paddle.disable_static()


class TestChooseShapeAttrOrApi(unittest.TestCase):
    def test_api_shape_is_none(self):
        self.assertEqual(
            paddle.jit.dy2static.choose_shape_attr_or_api([1, 2], None),
            [1, 2])
        self.assertEqual(
            paddle.jit.dy2static.choose_shape_attr_or_api([1], None), [1])
        self.assertEqual(
            paddle.jit.dy2static.choose_shape_attr_or_api([2, 3, 7], None, 0),
            2)

    def test_attr_shape_is_int(self):
        x = paddle.zeros([1, 3, 5, 7])
        self.assertEqual(
            paddle.jit.dy2static.choose_shape_attr_or_api(x.shape[0],
                                                          paddle.shape(x)[0]),
            1)
        self.assertEqual(
            paddle.jit.dy2static.choose_shape_attr_or_api(x.shape[1],
                                                          paddle.shape(x)[1]),
            3)
        self.assertEqual(
            paddle.jit.dy2static.choose_shape_attr_or_api(-1,
                                                          paddle.shape(x)[0]),
            paddle.shape(x)[0])
        self.assertEqual(
            paddle.jit.dy2static.choose_shape_attr_or_api(-1,
                                                          paddle.shape(x), 0),
            paddle.shape(x)[0])

    def test_positive_attr_shape(self):
        x = paddle.zeros([1, 3, 5, 7])
        self.assertEqual(
            paddle.jit.dy2static.choose_shape_attr_or_api(x.shape,
                                                          paddle.shape(x)),
            x.shape)
        self.assertEqual(
            paddle.jit.dy2static.choose_shape_attr_or_api(x.shape,
                                                          paddle.shape(x), 3),
            x.shape[3])

    def test_negative_attr_shape(self):
        x = paddle.zeros([7])
        self.assertEqual(
            paddle.jit.dy2static.choose_shape_attr_or_api([-1],
                                                          paddle.shape(x), 0),
            paddle.shape(x)[0])
        self.assertEqual(
            paddle.jit.dy2static.choose_shape_attr_or_api([-1],
                                                          paddle.shape(x)),
            paddle.shape(x))


class TestEvaIfExistElseNone(unittest.TestCase):
    def test_globals(self):
        global x_shape
        x_shape = [1, 2, 3]
        self.assertEqual(eval_if_exist_else_none('x_shape', locals()), None)
        self.assertEqual(eval_if_exist_else_none('x_shape', globals()), x_shape)

        del x_shape

    def test_enclosing_scope(self):
        global x_shape
        x_shape = [1, 2, 3]

        def foo():
            y_shape = [2, 3, 4]
            self.assertEqual(
                eval_if_exist_else_none('x_shape', globals()), [1, 2, 3])
            self.assertEqual(
                eval_if_exist_else_none('y_shape', locals()), [2, 3, 4])

        foo()
        del x_shape

    def test_global_in_func(self):
        x_shape = [1, 2, 3]

        def foo():
            global y_shape
            y_shape = [2, 3, 4]

            self.assertEqual(
                eval_if_exist_else_none('y_shape', globals()), [2, 3, 4])
            self.assertEqual(eval_if_exist_else_none('x_shape', locals()), None)
            self.assertEqual(
                eval_if_exist_else_none('x_shape', globals()), None)

            del y_shape

        foo()

    def test_none(self):
        def foo():
            x_shape = [2, 3, 4]
            return x_shape

        self.assertEqual(eval_if_exist_else_none('x_shape', locals()), None)


class ShapeLayer(paddle.nn.Layer):
    def __init__(self):
        super(ShapeLayer, self).__init__()

    @paddle.jit.to_static(input_spec=[paddle.static.InputSpec(shape=[None, 1])])
    def forward(self, x):
        x = paddle.reshape(x, [-1, x.shape[1]])
        bs = x.shape[0]  # -1

        # for trigger choos_shape_attr_or_api
        out = paddle.zeros([bs, 1], dtype='float32')
        return out


class TestChooseShapeAttrOrApiWithLayer(unittest.TestCase):
    def test_tensor_shape(self):
        x = paddle.zeros(shape=[4, 1], dtype='float32')
        net = ShapeLayer()
        out = net(x)

        self.assertTrue(np.array_equal(out.numpy(), x.numpy()))


if __name__ == '__main__':
    unittest.main()
