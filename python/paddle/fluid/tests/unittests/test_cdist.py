# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle import fluid
from paddle.fluid import core


def cdist(x, y, p):
    x_shape = x.shape
    y_shape = y.shape
    if len(x_shape) > len(y_shape):
        y_shape = (1,) * (len(x_shape) - len(y_shape)) + y_shape
    elif len(y_shape) > len(x_shape):
        x_shape = (1,) * (len(y_shape) - len(x_shape)) + x_shape
    resize_shape = ()
    for i in range(len(x_shape[:-2])):
        resize_shape = resize_shape + (max(x_shape[i], y_shape[i]),)
    new_y = np.concatenate([y] * x_shape[-2], axis=-2)
    new_x = np.repeat(x, y_shape[-2], axis=-2)
    if p == 0:
        loss = np.sum(np.abs(new_x - new_y) ** p, axis=-1)
    elif p == float('inf'):
        loss = np.max(np.abs(new_x - new_y), axis=-1)
    else:
        loss = np.sum(np.abs(new_x - new_y) ** p, axis=-1) ** (1 / p)
    loss = loss.reshape((*resize_shape, x_shape[-2], y_shape[-2]))
    return loss


class TestDistAPI(unittest.TestCase):
    def init_data_type(self):
        self.data_type = (
            'float32' if core.is_compiled_with_rocm() else 'float64'
        )

    def test_static(self):
        self.init_data_type()
        paddle.enable_static()
        main_program = fluid.Program()
        startup_program = fluid.Program()

        with fluid.program_guard(main_program, startup_program):
            x = paddle.static.data(
                name='x', shape=[2, 3, 4, 5], dtype=self.data_type
            )
            y = paddle.static.data(
                name='y', shape=[2, 3, 1, 5], dtype=self.data_type
            )
            x_i = np.random.random((2, 3, 4, 5)).astype(self.data_type)
            y_i = np.random.random((2, 3, 1, 5)).astype(self.data_type)
            p = 2
            result = paddle.cdist(x, y, p)
            place = (
                fluid.CUDAPlace(0)
                if core.is_compiled_with_cuda()
                else fluid.CPUPlace()
            )
            exe = fluid.Executor(place)
            out = exe.run(
                fluid.default_main_program(),
                feed={'x': x_i, 'y': y_i},
                fetch_list=[result],
            )
            np.testing.assert_allclose(cdist(x_i, y_i, p), out[0], rtol=1e-05)
        paddle.disable_static()

    def test_p_order(self):
        self.init_data_type()

        for p in [0, 2, float('inf'), 1.5]:
            a_i = np.random.random((2, 3, 4, 5)).astype(self.data_type)
            b_i = np.random.random((2, 3, 1, 5)).astype(self.data_type)
            a = paddle.to_tensor(a_i)
            b = paddle.to_tensor(b_i)
            c = paddle.cdist(a, b, p)
            np.testing.assert_allclose(
                cdist(a_i, b_i, p), c.numpy(), rtol=1e-05
            )

    def test_shape1(self):
        self.init_data_type()
        p = 2
        x_shape = (5, 6, 2)
        y_shape = (1, 5, 2)

        a_i = np.random.random(x_shape).astype(self.data_type)
        b_i = np.random.random(y_shape).astype(self.data_type)
        a = paddle.to_tensor(a_i)
        b = paddle.to_tensor(b_i)
        c = paddle.cdist(a, b, p)
        np.testing.assert_allclose(cdist(a_i, b_i, p), c.numpy(), rtol=1e-05)

    def test_shape2(self):
        self.init_data_type()
        p = 2
        x_shape = (4, 4, 3, 1)
        y_shape = (1, 5, 1)
        a_i = np.random.random(x_shape).astype(self.data_type)
        b_i = np.random.random(y_shape).astype(self.data_type)
        a = paddle.to_tensor(a_i)
        b = paddle.to_tensor(b_i)
        c = paddle.cdist(a, b, p)
        np.testing.assert_allclose(cdist(a_i, b_i, p), c.numpy(), rtol=1e-05)

    def test_shape3(self):
        self.init_data_type()
        p = 2
        x_shape = (2, 5)
        y_shape = (3, 1, 5, 5)
        a_i = np.random.random(x_shape).astype(self.data_type)
        b_i = np.random.random(y_shape).astype(self.data_type)
        a = paddle.to_tensor(a_i)
        b = paddle.to_tensor(b_i)
        c = paddle.cdist(a, b, p)
        np.testing.assert_allclose(cdist(a_i, b_i, p), c.numpy(), rtol=1e-05)

    def test_grad(self):
        a = paddle.rand([2, 2, 3, 2])
        b = paddle.rand([2, 2, 4, 2])
        a.stop_gradient = False
        c = paddle.cdist(a, b, 2)
        c.backward()

    def test_cdist_shape_error1(self):
        x = paddle.rand([2, 2, 3, 3])
        y = paddle.rand([2, 2, 4, 4])
        self.assertRaises(
            ValueError,
            paddle.cdist,
            x=x,
            y=y,
            p=2,
        )

    def test_cdist_shape_error2(self):
        x = paddle.rand([2, 4, 3, 3])
        y = paddle.rand([2, 2, 4, 3])
        self.assertRaises(
            ValueError,
            paddle.cdist,
            x=x,
            y=y,
            p=2,
        )

    def test_cdist_shape_error3(self):
        x = paddle.rand([3])
        y = paddle.rand([3])
        self.assertRaises(
            ValueError,
            paddle.cdist,
            x=x,
            y=y,
            p=2,
        )

    def test_cdist_p_order_error(self):
        x = paddle.rand([2, 4, 3, 3])
        y = paddle.rand([2, 4, 4, 3])
        self.assertRaises(
            ValueError,
            paddle.cdist,
            x=x,
            y=y,
            p=-3,
        )


if __name__ == '__main__':
    unittest.main()
