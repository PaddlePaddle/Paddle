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

from __future__ import print_function

import unittest
import numpy as np

import paddle.fluid as fluid
import paddle.fluid.dygraph as dg


class TestWeightNormWrapper(unittest.TestCase):
    def _test_case_for_linear(self, place):
        in_features = 16
        out_features = 32
        batch_size = 8

        x_np = np.random.randn(batch_size, in_features).astype(np.float64)

        with dg.guard(place):
            lin = dg.Linear(in_features, out_features, dtype="float64")
            x_var = dg.to_variable(x_np)
            y_var_original = lin(x_var)
            y_np_original = y_var_original.numpy()

            lin_wn = dg.WeightNormWrapper(lin, dim=-1)
            y_var_wn = lin_wn(x_var)
            y_np_wn = y_var_wn.numpy()

            self.assertTrue(np.allclose(y_np_original, y_np_wn))
            self.assertTrue(lin_wn.weight_g.shape == [out_features])

    def test_case_for_linear(self):
        self._test_case_for_linear(fluid.CPUPlace())
        if fluid.is_compiled_with_cuda():
            self._test_case_for_linear(fluid.CUDAPlace(0))

    def _test_case_for_conv2d(self, place):
        in_features = 16
        out_features = 16
        filter_size = (3, 3)
        spatial_shape = (32, 32)
        batch_size = 8

        x_np = np.random.randn(batch_size, in_features,
                               *spatial_shape).astype(np.float64)

        with dg.guard(place):
            conv = dg.Conv2D(
                in_features, out_features, filter_size, dtype="float64")
            x_var = dg.to_variable(x_np)
            y_var_original = conv(x_var)
            y_np_original = y_var_original.numpy()

            conv_wn = dg.WeightNormWrapper(conv, dim=0)
            y_var_wn = conv_wn(x_var)
            y_np_wn = y_var_wn.numpy()

            self.assertIs(
                getattr(conv_wn, "_filter_size"),
                getattr(conv_wn.layer, "_filter_size"),
                "WeightNormWrapper's {} is not it's layer's {}".format(
                    "_filter_size", "_filter_size"))
            self.assertTrue(np.allclose(y_np_original, y_np_wn))
            self.assertEqual(conv_wn.weight_g.shape, [out_features])

    def test_case_for_conv2d(self):
        self._test_case_for_conv2d(fluid.CPUPlace())
        if fluid.is_compiled_with_cuda():
            self._test_case_for_conv2d(fluid.CUDAPlace(0))

    def _test_case_for_conv2d_transpose(self, place):
        in_features = 16
        out_features = 16
        filter_size = (3, 3)
        spatial_shape = (32, 32)
        batch_size = 8

        x_np = np.random.randn(batch_size, in_features,
                               *spatial_shape).astype(np.float64)

        with dg.guard(place):
            conv = dg.Conv2DTranspose(
                in_features, out_features, filter_size, dtype="float64")
            x_var = dg.to_variable(x_np)
            y_var_original = conv(x_var)
            y_np_original = y_var_original.numpy()

            conv_wn = dg.WeightNormWrapper(conv, dim=1)
            y_var_wn = conv_wn(x_var)
            y_np_wn = y_var_wn.numpy()

            self.assertIs(
                getattr(conv_wn, "_filter_size"),
                getattr(conv_wn.layer, "_filter_size"),
                "WeightNormWrapper's {} is not it's layer's {}".format(
                    "_filter_size", "_filter_size"))
            self.assertTrue(np.allclose(y_np_original, y_np_wn))
            self.assertEqual(conv_wn.weight_g.shape, [out_features])

    def test_case_for_conv2d_transpose(self):
        self._test_case_for_conv2d_transpose(fluid.CPUPlace())
        if fluid.is_compiled_with_cuda():
            self._test_case_for_conv2d_transpose(fluid.CUDAPlace(0))


if __name__ == '__main__':
    unittest.main()
