# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

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
import paddle.nn.functional as F


class TestInterpolateAntialias(unittest.TestCase):
    def setUp(self):
        self.input_shape = (1, 1, 8, 8)
        self.input_data = paddle.arange(64, dtype="float32").reshape(
            self.input_shape
        )
        # A pattern that has high frequency components
        self.input_data[0, 0, ::2, ::2] = 100.0

    def test_bilinear_antialias(self):
        # Downsample by 0.5
        scale = 0.5
        out_no_aa = F.interpolate(
            self.input_data,
            scale_factor=scale,
            mode='bilinear',
            align_corners=False,
            antialias=False,
        )
        out_aa = F.interpolate(
            self.input_data,
            scale_factor=scale,
            mode='bilinear',
            align_corners=False,
            antialias=True,
        )

        # Results should be different
        self.assertFalse(
            np.allclose(out_no_aa.numpy(), out_aa.numpy()),
            "Bilinear: Antialias=True should differ from False",
        )
        print(
            "Bilinear Antialias test passed: Output differs from non-antialias."
        )

    def test_bicubic_antialias(self):
        # Downsample by 0.5
        scale = 0.5
        out_no_aa = F.interpolate(
            self.input_data,
            scale_factor=scale,
            mode='bicubic',
            align_corners=False,
            antialias=False,
        )
        out_aa = F.interpolate(
            self.input_data,
            scale_factor=scale,
            mode='bicubic',
            align_corners=False,
            antialias=True,
        )

        # Results should be different
        self.assertFalse(
            np.allclose(out_no_aa.numpy(), out_aa.numpy()),
            "Bicubic: Antialias=True should differ from False",
        )
        print(
            "Bicubic Antialias test passed: Output differs from non-antialias."
        )

    def test_error_on_other_modes(self):
        with self.assertRaises(ValueError):
            F.interpolate(
                self.input_data,
                scale_factor=0.5,
                mode='nearest',
                antialias=True,
            )

        with self.assertRaises(ValueError):
            F.interpolate(
                self.input_data, scale_factor=0.5, mode='linear', antialias=True
            )
        print("Error check passed: ValueError raised for unsupported modes.")

    def test_bilinear_antialias_grad(self):
        x = paddle.to_tensor(self.input_data, stop_gradient=False)
        scale = 0.5
        out = F.interpolate(
            x,
            scale_factor=scale,
            mode='bilinear',
            align_corners=False,
            antialias=True,
        )
        loss = out.mean()
        loss.backward()
        self.assertIsNotNone(x.grad)
        # Check if grad is not all zeros (it shouldn't be)
        self.assertTrue(np.any(x.grad.numpy() != 0))
        print("Bilinear Gradient test passed.")

    def test_bicubic_antialias_grad(self):
        x = paddle.to_tensor(self.input_data, stop_gradient=False)
        scale = 0.5
        out = F.interpolate(
            x,
            scale_factor=scale,
            mode='bicubic',
            align_corners=False,
            antialias=True,
        )
        loss = out.mean()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(np.any(x.grad.numpy() != 0))
        print("Bicubic Gradient test passed.")


if __name__ == '__main__':
    unittest.main()
