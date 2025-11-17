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
from op_test import get_device_place, is_custom_device

import paddle


class TestInterpolateParam(unittest.TestCase):
    def setUp(self):
        self.input_data = paddle.randn(shape=(2, 3, 6, 10)).astype(
            paddle.float32
        )

    def test_alias_input_for_x(self):
        """test parameter alias input/x"""
        out_with_input = paddle.nn.functional.interpolate(
            input=self.input_data, scale_factor=[2, 1], mode="bilinear"
        )
        out_with_x = paddle.nn.functional.interpolate(
            x=self.input_data, scale_factor=[2, 1], mode="bilinear"
        )

        np.testing.assert_array_equal(
            out_with_input.numpy(), out_with_x.numpy()
        )

    def test_params_consistency(self):
        """test both paddle and torch formats works."""
        out_torch = paddle.nn.functional.interpolate(
            self.input_data,  # input
            None,  # size
            [2, 1],  # scale_factor
            'bilinear',  # mode
            True,  # align_corners
            True,  # recompute_scale_factor
            False,  # antialias
        )

        out_paddle = paddle.nn.functional.interpolate(
            x=self.input_data,
            size=None,
            scale_factor=[2, 1],
            mode='bilinear',
            align_corners=True,
            recompute_scale_factor=True,
        )

        np.testing.assert_array_equal(out_torch.numpy(), out_paddle.numpy())

    def test_params_1(self):
        """test all args with torch format"""
        try:
            out_torch = paddle.nn.functional.interpolate(
                self.input_data,  # input
                None,  # size
                [2, 1],  # scale_factor
                'bilinear',  # mode
                True,  # align_corners
                True,  # recompute_scale_factor
                False,  # antialias
            )
            self.assertTrue(True, "Function call succeeded without error")
        except Exception as e:
            self.fail(f"Function raised an unexpected exception: {e}")

    def test_params_2(self):
        """test all kwargs with torch format"""
        try:
            out_torch = paddle.nn.functional.interpolate(
                input=self.input_data,
                size=None,
                scale_factor=[2, 1],
                mode='bilinear',
                align_corners=True,
                recompute_scale_factor=True,
                antialias=False,
            )
            self.assertTrue(True, "Function call succeeded without error")
        except Exception as e:
            self.fail(f"Function raised an unexpected exception: {e}")

    def test_params_3(self):
        """test of passing both args and kwargs parameters"""
        try:
            out1 = paddle.nn.functional.interpolate(
                input=self.input_data,
                size=None,
                scale_factor=[2, 1],
                mode='bilinear',
                align_corners=True,
                recompute_scale_factor=True,
                antialias=False,
            )
            out2 = paddle.nn.functional.interpolate(
                self.input_data,
                None,
                [2, 1],
                mode='bilinear',
                align_corners=True,
                recompute_scale_factor=True,
                antialias=False,
            )
            self.assertTrue(True, "Function call succeeded without error")
        except Exception as e:
            self.fail(f"Function raised an unexpected exception: {e}")

    def test_params_4(self):
        """test duplicate parameters"""
        with self.assertRaises(TypeError):
            out1 = paddle.nn.functional.interpolate(
                x=self.input_data,
                input=self.input_data,
                size=[12, 12],
            )
        with self.assertRaises(TypeError):
            out1 = paddle.nn.functional.interpolate(
                self.input_data,
                input=self.input_data,
                size=[12, 12],
            )

    def test_unsupported_antialias(self):
        """test unsupported antialias"""
        with self.assertRaises(TypeError):
            out1 = paddle.nn.functional.interpolate(
                x=self.input_data,
                input=self.input_data,
                size=[12, 12],
                antialias="True",
            )
        with self.assertRaises(ValueError):
            out1 = paddle.nn.functional.interpolate(
                self.input_data,
                input=self.input_data,
                size=[12, 12],
                mode="nearest",
                antialias=True,
            )
        with self.assertRaises(ValueError):
            paddle.enable_static()

            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data(
                    name='x', shape=[2, 3, 6, 10], dtype='float32'
                )
                out1 = paddle.nn.functional.interpolate(
                    self.input_data,
                    input=self.input_data,
                    size=[12, 12],
                    mode="nearest",
                    antialias=True,
                )

                place = (
                    get_device_place()
                    if (paddle.is_compiled_with_cuda() or is_custom_device())
                    else paddle.CPUPlace()
                )
                exe = paddle.static.Executor(place)

                input_data = np.random.rand(2, 3, 6, 10).astype('float32')
                feed = {'x': input_data}
                results = exe.run(feed=feed, fetch_list=[out1])

            paddle.disable_static()


if __name__ == '__main__':
    unittest.main()
