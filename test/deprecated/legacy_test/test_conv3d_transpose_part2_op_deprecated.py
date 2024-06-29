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

import sys
import unittest

import numpy as np

sys.path.append("../../legacy_test")

import paddle
from paddle import base
from paddle.base import core

paddle.enable_static()


class TestConv3DTransposeAPI(unittest.TestCase):
    def test_case1(self):
        data1 = paddle.static.data(
            name='data1', shape=[-1, 3, 5, 5, 5], dtype='float32'
        )
        data2 = paddle.static.data(
            name='data2', shape=[-1, 5, 5, 5, 3], dtype='float32'
        )

        out1 = paddle.static.nn.conv3d_transpose(
            input=data1,
            groups=1,
            num_filters=6,
            filter_size=3,
            data_format='NCDHW',
        )
        out2 = paddle.static.nn.conv3d_transpose(
            input=data2,
            groups=1,
            num_filters=6,
            filter_size=3,
            data_format='NDHWC',
        )
        out3 = paddle.static.nn.conv3d_transpose(
            input=data1,
            groups=1,
            num_filters=6,
            filter_size=3,
            padding=[[0, 0], [0, 0], [1, 1], [0, 0], [1, 1]],
            data_format='NCDHW',
        )
        out4 = paddle.static.nn.conv3d_transpose(
            input=data2,
            groups=3,
            num_filters=6,
            filter_size=3,
            padding=[[0, 0], [0, 0], [1, 1], [1, 2], [0, 0]],
            data_format='NDHWC',
        )
        out5 = paddle.static.nn.conv3d_transpose(
            input=data2,
            groups=1,
            num_filters=6,
            filter_size=3,
            padding='SAME',
            data_format='NCDHW',
        )
        out6 = paddle.static.nn.conv3d_transpose(
            input=data2,
            groups=1,
            num_filters=6,
            filter_size=3,
            padding='VALID',
            data_format='NDHWC',
        )
        out7 = paddle.static.nn.conv3d_transpose(
            input=data2,
            groups=1,
            num_filters=6,
            output_size=[7, 7, 7],
            padding=[0, 0, 0],
            data_format='NDHWC',
        )

        data1_np = np.random.random((2, 3, 5, 5, 5)).astype("float32")
        data2_np = np.random.random((2, 5, 5, 5, 3)).astype("float32")

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        exe = base.Executor(place)
        exe.run(base.default_startup_program())
        results = exe.run(
            base.default_main_program(),
            feed={"data1": data1_np, "data2": data2_np},
            fetch_list=[out1, out2, out3, out4, out5, out6, out7],
            return_numpy=True,
        )
        self.assertIsNotNone(results[0])
        self.assertIsNotNone(results[1])
        self.assertIsNotNone(results[2])
        self.assertIsNotNone(results[3])
        self.assertIsNotNone(results[4])
        self.assertIsNotNone(results[5])
        self.assertIsNotNone(results[6])


class TestConv3DTransposeOpException(unittest.TestCase):
    def test_exception(self):
        data = paddle.static.data(
            name='data', shape=[-1, 3, 5, 5, 5], dtype="float32"
        )

        def attr_data_format():
            out = paddle.static.nn.conv2d_transpose(
                input=data,
                groups=1,
                num_filters=6,
                filter_size=3,
                data_format="NCDW",
            )

        self.assertRaises(ValueError, attr_data_format)

        def attr_padding_str():
            out = paddle.static.nn.conv2d_transpose(
                input=data,
                groups=1,
                num_filters=6,
                filter_size=3,
                padding='Vald',
            )

        self.assertRaises(ValueError, attr_padding_str)

        def attr_padding_list():
            out = paddle.static.nn.conv2d_transpose(
                input=data,
                groups=1,
                num_filters=6,
                filter_size=3,
                padding=[[1, 1], [1, 1], [0, 0], [0, 0], [1, 1]],
            )

        self.assertRaises(ValueError, attr_padding_list)

        def attr_padding_with_data_format():
            out = paddle.static.nn.conv2d_transpose(
                input=data,
                groups=1,
                num_filters=6,
                filter_size=3,
                padding=[[1, 1], [0, 0], [0, 0], [1, 0], [1, 1]],
                data_format='NDHWC',
            )

        self.assertRaises(ValueError, attr_padding_with_data_format)


if __name__ == '__main__':
    unittest.main()
