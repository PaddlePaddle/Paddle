#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import to_tensor
from paddle.nn import ZeroPad1D


class TestZeroPad1dAPI(unittest.TestCase):
    def setUp(self):
        if paddle.is_compiled_with_cuda():
            paddle.device.set_device('gpu:0')
        else:
            paddle.device.set_device('cpu')
        self.shape = [4, 6, 6]
        self.support_dtypes = ['float32', 'float64', 'int32', 'int64']

    def test_support_dtypes(self):
        for dtype in self.support_dtypes:
            pad = 2
            x = np.random.randint(-255, 255, size=self.shape).astype(dtype)
            expect_res = np.pad(
                x,
                [[0, 0], [0, 0], [pad, pad]],
                mode='constant',
                constant_values=0,
            )

            x_tensor = to_tensor(x).astype(dtype)
            zeropad1d = ZeroPad1D(padding=pad)
            ret_res = zeropad1d(x_tensor).numpy()
            np.testing.assert_allclose(expect_res, ret_res, rtol=1e-05)

    def test_support_pad2(self):
        pad = [1, 2]
        x = np.random.randint(-255, 255, size=self.shape)
        expect_res = np.pad(
            x, [[0, 0], [0, 0], pad], mode='constant', constant_values=0
        )

        x_tensor = to_tensor(x)
        zeropad1d = ZeroPad1D(padding=pad)
        ret_res = zeropad1d(x_tensor).numpy()
        np.testing.assert_allclose(expect_res, ret_res, rtol=1e-05)

    def test_support_pad3(self):
        pad = (1, 2)
        x = np.random.randint(-255, 255, size=self.shape)
        expect_res = np.pad(x, [[0, 0], [0, 0], [pad[0], pad[1]]])

        x_tensor = to_tensor(x)
        zeropad1d = ZeroPad1D(padding=pad)
        ret_res = zeropad1d(x_tensor).numpy()
        np.testing.assert_allclose(expect_res, ret_res, rtol=1e-05)

    def test_support_pad4(self):
        pad = [1, 2]
        x = np.random.randint(-255, 255, size=self.shape)
        expect_res = np.pad(x, [[0, 0], [0, 0], [pad[0], pad[1]]])

        x_tensor = to_tensor(x)
        pad_tensor = to_tensor(pad, dtype='int32')
        zeropad1d = ZeroPad1D(padding=pad_tensor)
        ret_res = zeropad1d(x_tensor).numpy()
        np.testing.assert_allclose(expect_res, ret_res, rtol=1e-05)

    def test_repr(self):
        pad = [1, 2]
        zeropad1d = ZeroPad1D(padding=pad)
        name_str = zeropad1d.extra_repr()
        assert name_str == 'padding=[1, 2], data_format=NCL'


if __name__ == '__main__':
    unittest.main()
