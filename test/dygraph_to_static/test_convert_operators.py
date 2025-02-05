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

import unittest

import numpy as np
from dygraph_to_static_utils import (
    Dy2StTestBase,
    test_ast_only,
)

import paddle


class CallNotExist(paddle.nn.Layer):
    def __call__(self):
        # call a non-exist API to trigger exception
        return paddle.nn.not_exist_api


class CallableList(list):
    def __call__(self, x):
        return x


class ForwardNotExist(paddle.nn.Layer):
    def forward(self):
        return 0


net = ForwardNotExist()
net.forward = "A string so that convert forward will fail"


class TestConvertCall(Dy2StTestBase):
    # fallback mode will raise a InnerError, it's ok.
    @test_ast_only
    def test_class_exception(self):
        def call_not_exist():
            net = CallNotExist()
            return net()

        with self.assertRaises(AttributeError):
            paddle.jit.to_static(call_not_exist())

        def forward_not_exist():
            return net()

        with self.assertRaises(AttributeError):
            paddle.jit.to_static(forward_not_exist)()

    def test_callable_list(self):
        def callable_list(x, y):
            callable_list = CallableList()
            return callable_list(x) + y

        self.assertEqual(paddle.jit.to_static(callable_list)(1, 2), 3)


class ShapeLayer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = paddle.reshape(x, [-1, x.shape[1]])
        bs = x.shape[0]  # -1

        # for trigger choose_shape_attr_or_api
        out = paddle.zeros([bs, 1], dtype='float32')
        return out


class TestChooseShapeAttrOrApiWithLayer(Dy2StTestBase):
    def test_tensor_shape(self):
        x = paddle.zeros(shape=[4, 1], dtype='float32')
        net = paddle.jit.to_static(
            function=ShapeLayer(),
            input_spec=[paddle.static.InputSpec(shape=[None, 1])],
        )
        out = net(x)

        np.testing.assert_array_equal(out.numpy(), x.numpy())


class TestIfElseNoValue(Dy2StTestBase):
    def test_else_ret_none(self):
        input_x = paddle.to_tensor([[1, 2, 3], [4, 5, 6]])

        def with_common_value(x, use_cache=False):
            if use_cache:
                y = x + 1
                z = x + 2
                return y, z
            else:
                c = x + 1
                z = x - 1
                return None

        def without_common_value(x, use_cache=False):
            if use_cache:
                y = x + 1
                z = x + 2
                return y, z
            else:
                c = x + 1
                return None

        out = paddle.jit.to_static(with_common_value)(input_x, False)
        self.assertIsNone(out)
        out = paddle.jit.to_static(without_common_value)(input_x, False)
        self.assertIsNone(out)

    def test_else_ret_c(self):
        input_x = paddle.to_tensor([[1, 2, 3], [4, 5, 6]])

        def with_common_value(x, use_cache=False):
            if use_cache:
                y = x + 1
                z = x + 2
                return y, z
            else:
                c = x + 1
                z = x - 1
                return c

        def without_common_value(x, use_cache=False):
            if use_cache:
                y = x + 1
                z = x + 2
                return y, z
            else:
                c = x + 1
                return c

        out = paddle.jit.to_static(with_common_value)(input_x, False)
        self.assertListEqual(paddle.tolist(out), paddle.tolist(input_x + 1))
        out = paddle.jit.to_static(without_common_value)(input_x, False)
        self.assertListEqual(paddle.tolist(out), paddle.tolist(input_x + 1))
        y, z = paddle.jit.to_static(with_common_value)(input_x, True)
        self.assertListEqual(paddle.tolist(y), paddle.tolist(input_x + 1))
        self.assertListEqual(paddle.tolist(z), paddle.tolist(input_x + 2))

    def test_else_ret_cz(self):
        input_x = paddle.to_tensor([[1, 2, 3], [4, 5, 6]])

        def with_common_value(x, use_cache=False):
            if use_cache:
                y = x + 1
                z = x + 2
                return y, z, 1
            else:
                c = x + 1
                z = x - 1
                return c, z

        def without_common_value(x, use_cache=False):
            if use_cache:
                y = x + 1
                z = x + 2
                return y, z, 1
            else:
                c = x + 1
                d = x - 1
                return c, d

        c, z = paddle.jit.to_static(with_common_value)(input_x, False)
        self.assertListEqual(paddle.tolist(c), paddle.tolist(input_x + 1))
        self.assertListEqual(paddle.tolist(z), paddle.tolist(input_x - 1))
        c, d = paddle.jit.to_static(without_common_value)(input_x, False)
        self.assertListEqual(paddle.tolist(c), paddle.tolist(input_x + 1))
        self.assertListEqual(paddle.tolist(d), paddle.tolist(input_x - 1))


if __name__ == '__main__':
    unittest.main()
