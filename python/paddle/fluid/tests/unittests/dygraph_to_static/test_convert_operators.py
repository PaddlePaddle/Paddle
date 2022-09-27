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

        with self.assertRaises(AttributeError):
            forward_not_exist()


class TestConvertShapeCompare(unittest.TestCase):

    def test_non_variable(self):
        self.assertEqual(paddle.jit.dy2static.convert_shape_compare(1, "<", 2),
                         True)
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
            paddle.jit.dy2static.convert_shape_compare(1, ">", 2, "<=",
                                                       lambda: error_func()),
            False)

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
                paddle.jit.dy2static.convert_shape_compare(
                    x, "is", x, "is not", y), True)
            self.assertEqual(
                paddle.jit.dy2static.convert_shape_compare(
                    x, "is not", x, "is not", y), False)
            self.assertEqual(
                paddle.jit.dy2static.convert_shape_compare(x, "is", x, "is", y),
                False)

            eq_out = paddle.jit.dy2static.convert_shape_compare(x, "==", y)
            not_eq_out = paddle.jit.dy2static.convert_shape_compare(x, "!=", y)
            long_eq_out = paddle.jit.dy2static.convert_shape_compare(
                x, "==", x, "!=", y)

            place = paddle.CUDAPlace(
                0) if paddle.is_compiled_with_cuda() else paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            x_y_eq_out = exe.run(feed={
                "x": np.ones([3, 2]).astype(np.float32),
                "y": np.ones([3, 2]).astype(np.float32)
            },
                                 fetch_list=[eq_out, not_eq_out, long_eq_out])
            np.testing.assert_array_equal(np.array(x_y_eq_out),
                                          np.array([[True], [False], [False]]))

            set_a_zero = np.ones([3, 2]).astype(np.float32)
            set_a_zero[0][0] = 0.0
            x_y_not_eq_out = exe.run(
                feed={
                    "x": np.ones([3, 2]).astype(np.float32),
                    "y": set_a_zero
                },
                fetch_list=[eq_out, not_eq_out, long_eq_out])
            np.testing.assert_array_equal(np.array(x_y_not_eq_out),
                                          np.array([[False], [True], [True]]))
        paddle.disable_static()


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

        np.testing.assert_array_equal(out.numpy(), x.numpy())


class TestIfElseNoValue(unittest.TestCase):

    def test_else_ret_none(self):
        input_x = paddle.to_tensor([[1, 2, 3], [4, 5, 6]])

        @paddle.jit.to_static
        def with_common_value(x, use_cache=False):
            if use_cache:
                y = x + 1
                z = x + 2
                return y, z
            else:
                c = x + 1
                z = x - 1
                return None

        @paddle.jit.to_static
        def without_common_value(x, use_cache=False):
            if use_cache:
                y = x + 1
                z = x + 2
                return y, z
            else:
                c = x + 1
                return None

        out = with_common_value(input_x, False)
        self.assertIsNone(out)
        out = without_common_value(input_x, False)
        self.assertIsNone(out)

    def test_else_ret_c(self):
        input_x = paddle.to_tensor([[1, 2, 3], [4, 5, 6]])

        @paddle.jit.to_static
        def with_common_value(x, use_cache=False):
            if use_cache:
                y = x + 1
                z = x + 2
                return y, z
            else:
                c = x + 1
                z = x - 1
                return c

        @paddle.jit.to_static
        def without_common_value(x, use_cache=False):
            if use_cache:
                y = x + 1
                z = x + 2
                return y, z
            else:
                c = x + 1
                return c

        out = with_common_value(input_x, False)
        self.assertListEqual(paddle.tolist(out), paddle.tolist(input_x + 1))
        out = without_common_value(input_x, False)
        self.assertListEqual(paddle.tolist(out), paddle.tolist(input_x + 1))
        y, z = with_common_value(input_x, True)
        self.assertListEqual(paddle.tolist(y), paddle.tolist(input_x + 1))
        self.assertListEqual(paddle.tolist(z), paddle.tolist(input_x + 2))

    def test_else_ret_cz(self):
        input_x = paddle.to_tensor([[1, 2, 3], [4, 5, 6]])

        @paddle.jit.to_static
        def with_common_value(x, use_cache=False):
            if use_cache:
                y = x + 1
                z = x + 2
                return y, z, 1
            else:
                c = x + 1
                z = x - 1
                return c, z

        @paddle.jit.to_static
        def without_common_value(x, use_cache=False):
            if use_cache:
                y = x + 1
                z = x + 2
                return y, z, 1
            else:
                c = x + 1
                d = x - 1
                return c, d

        c, z = with_common_value(input_x, False)
        self.assertListEqual(paddle.tolist(c), paddle.tolist(input_x + 1))
        self.assertListEqual(paddle.tolist(z), paddle.tolist(input_x - 1))
        c, d = without_common_value(input_x, False)
        self.assertListEqual(paddle.tolist(c), paddle.tolist(input_x + 1))
        self.assertListEqual(paddle.tolist(d), paddle.tolist(input_x - 1))


if __name__ == '__main__':
    unittest.main()
