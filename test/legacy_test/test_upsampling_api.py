#   Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


class TestUpsamplingBilinear2D_Compatibility(unittest.TestCase):
    def test_upsamplingbilinear2d(self):
        with paddle.base.dygraph.guard():
            input = paddle.rand(shape=(2, 3, 6, 10)).astype("float32")
            upsample_out = paddle.nn.UpsamplingBilinear2D(size=[12, 12])
            output = upsample_out(x=input)
            out_np = output.numpy()

            # test func alias
            upsample_out = paddle.nn.UpsamplingBilinear2d(size=[12, 12])
            output = upsample_out(x=input)
            np.testing.assert_allclose(out_np, output.numpy())

            # test @param_one_alias(["x", "input"])
            upsample_out = paddle.nn.UpsamplingBilinear2D(size=[12, 12])
            output = upsample_out(input=input)
            np.testing.assert_allclose(out_np, output.numpy())

            # test both
            upsample_out = paddle.nn.UpsamplingBilinear2d(size=[12, 12])
            output = upsample_out(input=input)
            np.testing.assert_allclose(out_np, output.numpy())


class TestUpsamplingNearest2D_Compatibility(unittest.TestCase):
    def test_upsamplingnearest2d(self):
        with paddle.base.dygraph.guard():
            input = paddle.rand(shape=(2, 3, 6, 10)).astype("float32")
            upsample_out = paddle.nn.UpsamplingNearest2D(size=[12, 12])
            output = upsample_out(x=input)
            out_np = output.numpy()

            # test func alias
            upsample_out = paddle.nn.UpsamplingNearest2d(size=[12, 12])
            output = upsample_out(x=input)
            np.testing.assert_allclose(out_np, output.numpy())

            # test @param_one_alias(["x", "input"])
            upsample_out = paddle.nn.UpsamplingNearest2D(size=[12, 12])
            output = upsample_out(input=input)
            np.testing.assert_allclose(out_np, output.numpy())

            # test both
            upsample_out = paddle.nn.UpsamplingNearest2d(size=[12, 12])
            output = upsample_out(input=input)
            np.testing.assert_allclose(out_np, output.numpy())


if __name__ == '__main__':
    unittest.main()
