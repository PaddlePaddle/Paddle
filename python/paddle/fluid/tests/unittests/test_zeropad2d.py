#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import to_tensor
from paddle.nn.functional import zeropad2d
from paddle.nn import ZeroPad2D


class TestZeroPad2dAPIError(unittest.TestCase):
    """
    test paddle.zeropad2d error.
    """

    def setUp(self):
        """
        unsupport dtypes
        """
        self.shape = [4, 3, 224, 224]
        self.unsupport_dtypes = ['bool', 'int8']

    def test_unsupport_dtypes(self):
        """
        test unsupport dtypes.
        """
        for dtype in self.unsupport_dtypes:
            pad = 2
            x = np.random.randint(-255, 255, size=self.shape)
            x_tensor = to_tensor(x).astype(dtype)
            self.assertRaises(TypeError, zeropad2d, x=x_tensor, padding=pad)


class TestZeroPad2dAPI(unittest.TestCase):
    """
    test paddle.zeropad2d
    """

    def setUp(self):
        """
        support dtypes
        """
        self.shape = [4, 3, 224, 224]
        self.support_dtypes = ['float32', 'float64', 'int32', 'int64']

    def test_support_dtypes(self):
        """
        test support types
        """
        for dtype in self.support_dtypes:
            pad = 2
            x = np.random.randint(-255, 255, size=self.shape).astype(dtype)
            expect_res = np.pad(x, [[0, 0], [0, 0], [pad, pad], [pad, pad]])

            x_tensor = to_tensor(x).astype(dtype)
            ret_res = zeropad2d(x_tensor, [pad, pad, pad, pad]).numpy()
            self.assertTrue(np.allclose(expect_res, ret_res))

    def test_support_pad2(self):
        """
        test the type of 'pad' is list.
        """
        pad = [1, 2, 3, 4]
        x = np.random.randint(-255, 255, size=self.shape)
        expect_res = np.pad(
            x, [[0, 0], [0, 0], [pad[2], pad[3]], [pad[0], pad[1]]])

        x_tensor = to_tensor(x)
        ret_res = zeropad2d(x_tensor, pad).numpy()
        self.assertTrue(np.allclose(expect_res, ret_res))

    def test_support_pad3(self):
        """
        test the type of 'pad' is tuple.
        """
        pad = (1, 2, 3, 4)
        x = np.random.randint(-255, 255, size=self.shape)
        expect_res = np.pad(
            x, [[0, 0], [0, 0], [pad[2], pad[3]], [pad[0], pad[1]]])

        x_tensor = to_tensor(x)
        ret_res = zeropad2d(x_tensor, pad).numpy()
        self.assertTrue(np.allclose(expect_res, ret_res))

    def test_support_pad4(self):
        """
        test the type of 'pad' is paddle.Tensor.
        """
        pad = [1, 2, 3, 4]
        x = np.random.randint(-255, 255, size=self.shape)
        expect_res = np.pad(
            x, [[0, 0], [0, 0], [pad[2], pad[3]], [pad[0], pad[1]]])

        x_tensor = to_tensor(x)
        pad_tensor = to_tensor(pad, dtype='int32')
        ret_res = zeropad2d(x_tensor, pad_tensor).numpy()
        self.assertTrue(np.allclose(expect_res, ret_res))


class TestZeroPad2DLayer(unittest.TestCase):
    """
    test nn.ZeroPad2D
    """

    def setUp(self):
        self.shape = [4, 3, 224, 224]
        self.pad = [2, 2, 4, 1]
        self.padLayer = ZeroPad2D(padding=self.pad)
        self.x = np.random.randint(-255, 255, size=self.shape)
        self.expect_res = np.pad(self.x,
                                 [[0, 0], [0, 0], [self.pad[2], self.pad[3]],
                                  [self.pad[0], self.pad[1]]])

    def test_layer(self):
        self.assertTrue(
            np.allclose(
                zeropad2d(to_tensor(self.x), self.pad).numpy(),
                self.padLayer(to_tensor(self.x))))


if __name__ == '__main__':
    unittest.main()
