# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import paddle.nn as nn


def upsample_2d(img, type='float64', scale_factor=None, size=[12, 12], data_format="NCHW"):
    if size == None:
        size = [img.shape[2] * scale_factor, img.shapep[3] * scale_factor]
    if data_format == 'NCHW':
        num_batches, channels, height, width = img.shape
        emptyImage = np.zeros((num_batches, channels, size[0], size[1]), type)
        sh = size[0] / height
        sw = size[1] / width
        for num_batche in range(num_batches):
            for channel in range(channels):
                for i in range(size[0]):
                    for j in range(size[1]):
                        x = int(i / sh)
                        y = int(j / sw)
                        emptyImage[num_batche, channel, i, j] = img[num_batche, channel, x, y]
    elif data_format == 'NHWC':
        img = img.transpose((0, 3, 1, 2))
        num_batches, channels, height, width = img.shape
        emptyImage = np.zeros((num_batches, channels, size[0], size[1]), type)
        sh = size[0] / height
        sw = size[1] / width
        for num_batche in range(num_batches):
            for channel in range(channels):
                for i in range(size[0]):
                    for j in range(size[1]):
                        x = int(i / sh)
                        y = int(j / sw)
                        emptyImage[num_batche, channel, i, j] = img[num_batche, channel, x, y]
        emptyImage = emptyImage.transpose((0, 2, 3, 1))
    return emptyImage


class TestUpsamplingNearest2D(unittest.TestCase):
    def test_input_range(self):
        # generate test data
        np.random.seed(0)
        input_data = paddle.rand(shape=[2, 3, 6, 10])
        # data_format='NCHW'
        # generation Function
        upsample = nn.UpsamplingNearest2D(size=[12, 12], data_format="NCHW")
        # result verification
        for i in range(-3, 10):
            times = 10 ** i
            x = paddle.to_tensor(input_data * times).astype('float64')
            res = upsample(x).numpy().flatten().tolist()
            res_ = upsample_2d(x).flatten().tolist()
            self.assertAlmostEqual(res, res_)

        # data_format='NHWC'
        # generation Function
        upsample = nn.UpsamplingNearest2D(size=[12, 12], data_format="NHWC")
        # result verification
        for i in range(-3, 10):
            times = 10 ** i
            x = paddle.to_tensor(input_data * times).astype('float64')
            res = upsample(x).numpy().flatten().tolist()
            res_ = upsample_2d(x, data_format="NHWC").flatten().tolist()
            self.assertAlmostEqual(res, res_)

    def test_input_dtype(self):
        # generate test data
        np.random.seed(0)
        input_data = paddle.rand(shape=[2, 3, 6, 10])

        # data_format='NCHW'
        # generation Function
        upsample = nn.UpsamplingNearest2D(size=[12, 12], data_format="NCHW")
        # Data type test
        type_list = ['float32', 'float64', 'int32', 'uint8','int64']
        for try_type in type_list:
            x = paddle.to_tensor(input_data).astype(try_type)
            result = upsample(x)

        # data_format='NHWC'
        # generation Function
        upsample = nn.UpsamplingNearest2D(size=[12, 12], data_format="NHWC")
        # Data type test
        type_list = ['float32', 'float64', 'int32', 'uint8','int64']
        for try_type in type_list:
            x = paddle.to_tensor(input_data).astype(try_type)
            result = upsample(x)

    def test_wrong_input_dtype(self):
        # generate test data
        np.random.seed(0)
        input_data = paddle.rand(shape=[2, 3, 6, 10])

        # data_format='NCHW'
        # generation Function
        upsample = nn.UpsamplingNearest2D(size=[12, 12], data_format="NCHW")
        # generation type list
        type_list = [
            'bool', 'float16', 'int8', 'int16'
        ]
        # Data type test
        for try_type in type_list:
            x = paddle.to_tensor(input_data).astype(try_type)
            self.assertRaises(RuntimeError, upsample, x)

        # data_format='NHWC'
        # generation Function
        upsample = nn.UpsamplingNearest2D(size=[12, 12], data_format="NHWC")
        # generation type list
        type_list = [
            'bool', 'float16', 'int8', 'int16'
        ]
        # Data type test
        for try_type in type_list:
            x = paddle.to_tensor(input_data).astype(try_type)
            self.assertRaises(RuntimeError, upsample, x)

    def test_range(self):
        # when x are larger than 10**9 or less than 10**-4, the result may error
        # generate test data
        np.random.seed(0)
        input_data = paddle.rand(shape=[2, 3, 6, 10])

        # Large number test

        # data_format='NCHW'
        # generation Function
        upsample = nn.UpsamplingNearest2D(size=[12, 12], data_format="NCHW")
        # data modification
        x = paddle.to_tensor(input_data * 10 ** 10).astype('float64')
        # result verification
        res = upsample(x).numpy().flatten().tolist()
        res_ = upsample_2d(x).flatten().tolist()
        self.assertEqual(res, res_)

        # data modification
        x = paddle.to_tensor(input_data * 10 ** -4).astype('float64')
        # data conversion
        res = upsample(x).numpy().flatten().tolist()
        res_ = upsample_2d(x).flatten().tolist()
        self.assertEqual(res, res_)

        # Small number test

        # data_format='NHWC'
        # generation Function
        upsample = nn.UpsamplingNearest2D(size=[12, 12], data_format="NHWC")
        # data modification
        x = paddle.to_tensor(input_data * 10 ** 10).astype('float64')
        # data conversion
        res = upsample(x).numpy().flatten().tolist()
        res_ = upsample_2d(x, data_format="NHWC").flatten().tolist()
        self.assertEqual(res, res_)

        # data modificatio
        x = paddle.to_tensor(input_data * 10 ** -4).astype('float64')
        # data conversion
        res = upsample(x).numpy().flatten().tolist()
        res_ = upsample_2d(x, data_format="NHWC").flatten().tolist()
        self.assertEqual(res, res_)


if __name__ == '__main__':
    unittest.main()
