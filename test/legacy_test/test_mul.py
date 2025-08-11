# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import get_device_place

import paddle
from paddle import static


class TestMulApi(unittest.TestCase):
    def setUp(self) -> None:
        self.shape = [2, 3]
        self.dtype = 'float32'
        self.place = get_device_place()

    def test_static_api(self):
        paddle.enable_static()
        x_np = np.random.rand(*self.shape).astype(self.dtype)
        other2_np = np.random.rand(*self.shape).astype(self.dtype)
        other3_np = np.random.rand(self.shape[0], 1).astype(self.dtype)
        with static.program_guard(static.Program()):
            x = paddle.static.data(name='x', shape=self.shape, dtype=self.dtype)
            # other1 = 3.0
            other2 = paddle.static.data(
                name='other', shape=self.shape, dtype=self.dtype
            )
            other3 = paddle.static.data(
                name='other3', shape=[self.shape[0], 1], dtype=self.dtype
            )
            # out1 = x.mul(other1)
            out2 = x.mul(other2)
            out3 = x.mul(other3)
            exe = static.Executor(self.place)
            outs = exe.run(
                feed={'x': x_np, 'other': other2_np, 'other3': other3_np},
                # fetch_list=[out1, out2, out3],
                fetch_list=[out2, out3],
            )
            # np.testing.assert_allclose(
            #     outs[0], np.multiply(x_np, other1), rtol=1e-05
            # )
            np.testing.assert_allclose(
                outs[0], np.multiply(x_np, other2_np), rtol=1e-05
            )
            np.testing.assert_allclose(
                outs[1], np.multiply(x_np, other3_np), rtol=1e-05
            )

    def test_dyn_api(self):
        paddle.disable_static()
        x_np = np.random.rand(*self.shape).astype(self.dtype)
        other2_np = np.random.rand(*self.shape).astype(self.dtype)
        other3_np = np.random.rand(self.shape[0], 1).astype(self.dtype)
        x = paddle.to_tensor(x_np, place=self.place)
        # other1 = 3.0
        other2 = paddle.to_tensor(other2_np, place=self.place)
        other3 = paddle.to_tensor(other3_np, place=self.place)

        # out1 = x.mul(other1)
        out2 = x.mul(other2)
        out3 = x.mul(other3)

        # np.testing.assert_allclose(
        #     out1.numpy(), np.multiply(x_np, other1), rtol=1e-05
        # )
        np.testing.assert_allclose(
            out2.numpy(), np.multiply(x_np, other2_np), rtol=1e-05
        )
        np.testing.assert_allclose(
            out3.numpy(), np.multiply(x_np, other3_np), rtol=1e-05
        )


class TestMulInplaceApi(unittest.TestCase):
    def setUp(self) -> None:
        self.shape = [2, 3]
        self.dtype = 'float32'

    def test_dyn_api(self):
        paddle.disable_static()
        others = [
            # 3.0,
            paddle.to_tensor(np.random.rand(*self.shape).astype('float32')),
            paddle.to_tensor(np.random.rand(*self.shape).astype('float32'))[
                :, -1
            ].unsqueeze(-1),
        ]
        for other in others:
            x_np = np.random.rand(*self.shape).astype('float32')
            x = paddle.to_tensor(x_np)
            x.mul_(other)
            np.testing.assert_allclose(
                x.numpy(),
                np.multiply(
                    x_np,
                    (
                        other.numpy()
                        if isinstance(other, paddle.Tensor)
                        else other
                    ),
                ),
                rtol=1e-05,
            )


class TestMulInplaceError(unittest.TestCase):
    def test_errors(self):
        paddle.disable_static()
        # test dynamic computation graph: inputs must be broadcastable
        x_data = np.random.rand(3, 4)
        y_data = np.random.rand(2, 3, 4)
        x = paddle.to_tensor(x_data)
        y = paddle.to_tensor(y_data)

        def multiply_shape_error():
            with paddle.no_grad():
                x.mul_(y)

        self.assertRaises(ValueError, multiply_shape_error)
        paddle.enable_static()


if __name__ == '__main__':
    unittest.main()
