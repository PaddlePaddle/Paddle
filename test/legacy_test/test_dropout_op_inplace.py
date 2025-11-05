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
    get_device_place,
    get_places,
    is_custom_device,
)

import paddle
from paddle import base
from paddle.base import core


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

    def test_inplace_axis(self):
        for place in self.places:
            with base.dygraph.guard(place):
                in_np = np.random.random([2, 3, 4]).astype("float32")
                input = paddle.to_tensor(in_np)
                input_id = id(input)
                result = paddle.nn.functional.dropout(
                    x=input, p=0.0, axis=0, training=True, inplace=True
                )
                self.assertEqual(id(result), input_id)

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


class TestDropoutInplacePIR(unittest.TestCase):
    def test_pir_mode(self):
        with (
            paddle.pir_utils.IrGuard(),
            paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ),
        ):
            input = paddle.static.data(
                name='x', shape=[32, 64], dtype='float32'
            )
            input_id = id(input)
            result = paddle.nn.functional.dropout(
                input, p=0.0, training=False, inplace=True
            )
            self.assertEqual(id(result), input_id)


class TestDropoutInplaceAxisDygraph(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.places = get_places()

    def test_inplace_axis_training(self):
        for place in self.places:
            with base.dygraph.guard(place):
                in_np = np.random.random([2, 3, 4]).astype("float32")
                input = paddle.to_tensor(in_np)
                input_id = id(input)
                result = paddle.nn.functional.dropout(
                    x=input,
                    p=0.5,
                    axis=1,
                    training=True,
                    mode='upscale_in_train',
                    inplace=True,
                )
                self.assertEqual(id(result), input_id)

    def test_inplace_axis_p_one_dynamic(self):
        for place in self.places:
            with base.dygraph.guard(place):
                if not isinstance(place, paddle.CPUPlace):
                    in_np = np.random.random([2, 3, 4]).astype("float32")
                    input = paddle.to_tensor(in_np)
                    input_id = id(input)
                    result = paddle.nn.functional.dropout(
                        x=input, p=1.0, axis=0, training=True, inplace=True
                    )
                    self.assertEqual(id(result), input_id)

    def test_inplace_axis_p_one_pir(self):
        with (
            paddle.pir_utils.IrGuard(),
            paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ),
        ):
            in_np = np.random.random([2, 3, 4]).astype("float32")
            input = paddle.to_tensor(in_np)
            input_id = id(input)
            result = paddle.nn.functional.dropout(
                x=input, p=1.0, axis=0, training=True, inplace=True
            )
            self.assertEqual(id(result), input_id)

    def test_inplace_downscale_axis(self):
        for place in self.places:
            with base.dygraph.guard(place):
                in_np = np.random.random([2, 3, 4]).astype("float32")
                input = paddle.to_tensor(in_np)
                input_id = id(input)
                result = paddle.nn.functional.dropout(
                    x=input,
                    p=0.5,
                    axis=[0, 1],
                    mode='downscale_in_infer',
                    training=False,
                    inplace=True,
                )
                self.assertEqual(id(result), input_id)


class TestDropoutInplaceFP16(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)

    @unittest.skipIf(
        not (core.is_compiled_with_cuda() or is_custom_device()),
        "core is not compiled with CUDA",
    )
    def test_inplace_fp16(self):
        place = get_device_place()
        with base.dygraph.guard(place):
            in_np = np.random.random([32, 64]).astype("float16")
            input = paddle.to_tensor(in_np)
            input_id = id(input)
            result = paddle.nn.functional.dropout(x=input, p=0.0, inplace=True)
            self.assertEqual(id(result), input_id)


class TestDropoutInplaceBF16(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)

    def test_inplace_bf16(self):
        for place in get_places():
            with base.dygraph.guard(place):
                in_np = np.random.random([32, 64]).astype("float32")
                input = paddle.to_tensor(in_np)
                input = paddle.cast(input, 'bfloat16')
                input_id = id(input)
                result = paddle.nn.functional.dropout(
                    x=input, p=0.0, inplace=True
                )
                self.assertEqual(id(result), input_id)


class TestDropoutLayerInplace(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
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

    def test_dropout_layer_extra_repr(self):
        m1 = paddle.nn.Dropout(p=0.5, inplace=True)
        self.assertIn('inplace=True', m1.extra_repr())


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
