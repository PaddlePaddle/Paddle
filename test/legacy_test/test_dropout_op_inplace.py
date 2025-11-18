#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import (
    get_device_class,
    get_places,
)

import paddle
from paddle import base


class TestDropoutInplaceDygraph(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.places = get_places()

    def test_inplace_dygraph(self):
        for place in self.places:
            with base.dygraph.guard(place):
                in_np = np.random.random([32, 64]).astype("float32")
                input = paddle.to_tensor(in_np)
                input_id = id(input)
                result = paddle.nn.functional.dropout(
                    x=input, p=0.0, inplace=True
                )
                self.assertEqual(id(result), input_id)
                np.testing.assert_allclose(result.numpy(), in_np, rtol=1e-05)

    def test_inplace_p_one(self):
        for place in self.places:
            with base.dygraph.guard(place):
                in_np = np.random.random([32, 64]).astype("float32")
                input = paddle.to_tensor(in_np)
                input_id = id(input)
                result = paddle.nn.functional.dropout(
                    x=input, p=1.0, training=True, inplace=True
                )
                self.assertEqual(id(result), input_id)
                np.testing.assert_allclose(
                    result.numpy(), np.zeros_like(in_np), rtol=1e-05
                )

    def test_inplace_downscale(self):
        for place in self.places:
            with base.dygraph.guard(place):
                in_np = np.random.random([32, 64]).astype("float32")
                input = paddle.to_tensor(in_np)
                input_id = id(input)
                result = paddle.nn.functional.dropout(
                    x=input,
                    p=0.5,
                    mode='downscale_in_infer',
                    training=False,
                    inplace=True,
                )
                self.assertEqual(id(result), input_id)


class DropoutNet(paddle.nn.Layer):
    def __init__(self, p=0.0, training=False, inplace=True):
        super().__init__()
        self.p = p
        self.training = training
        self.inplace = inplace

    def forward(self, x):
        return paddle.nn.functional.dropout(
            x, p=self.p, training=self.training, inplace=self.inplace
        )


class TestDropoutInplacePIR(unittest.TestCase):
    def setUp(self):
        self.places = get_places()
        self.device_class = get_device_class()

    def test_pir_mode(self):
        with (
            paddle.pir_utils.IrGuard(),
            paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ),
        ):
            input_data = paddle.static.data(
                name='x', shape=[32, 64], dtype='float32'
            )
            input_id = id(input_data)
            result = paddle.nn.functional.dropout(
                input_data, p=0.0, training=False, inplace=True
            )
            self.assertEqual(id(result), input_id)

    def test_pir_cinn_mode(self):
        for place in self.places:
            if not isinstance(place, self.device_class):
                continue

            paddle.disable_static()
            paddle.set_device(place)

            net = DropoutNet(p=0.5, training=False, inplace=True)
            try:
                static_net = paddle.jit.to_static(
                    net, backend="CINN", full_graph=True
                )
            except Exception as e:
                self.fail(
                    f"paddle.jit.to_static(backend='CINN') failed on {place}: {e}"
                )

            x_np = np.random.random((32, 64)).astype("float32")
            x = paddle.to_tensor(x_np, place=place)

            try:
                result = static_net(x)
            except Exception as e:
                self.fail(
                    f"Running CINN-compiled network failed on {place}: {e}"
                )

            np.testing.assert_allclose(
                result.numpy(),
                x_np,
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"CINN dropout(training=False) output mismatch on {place}.",
            )

        paddle.enable_static()


class TestDropoutInplaceAxisDygraph(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.places = get_places()

    def test_inplace_axis_error(self):
        for place in self.places:
            with (
                base.dygraph.guard(place),
                self.assertRaises(NotImplementedError),
            ):
                in_np = np.random.random([2, 3, 4]).astype("float32")
                input = paddle.to_tensor(in_np)
                result = paddle.nn.functional.dropout(
                    x=input,
                    p=0.5,
                    axis=1,
                    training=True,
                    inplace=True,
                )


class TestDropoutLayerInplace(unittest.TestCase):
    def setUp(self):
        np.random.seed(0x0721)
        self.places = get_places()

    def test_dropout_layer_inplace(self):
        for place in self.places:
            with base.dygraph.guard(place):
                in_np = np.random.random([32, 64]).astype("float32")
                input = paddle.to_tensor(in_np)
                input_id = id(input)
                m = paddle.nn.Dropout(p=0.0, inplace=True)
                m.eval()
                result = m(input)
                self.assertEqual(id(result), input_id)
                np.testing.assert_allclose(result.numpy(), in_np, rtol=1e-05)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
