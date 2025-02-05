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

import paddle
from paddle import base
from paddle.base import core


class TestBilinearAPI(unittest.TestCase):

    def test_api(self):
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(startup, main):
            if core.is_compiled_with_cuda():
                place = core.CUDAPlace(0)
            else:
                place = core.CPUPlace()
            exe = base.Executor(place)

            data1 = paddle.static.data(name='X1', shape=[5, 5], dtype='float32')
            data2 = paddle.static.data(name='X2', shape=[5, 4], dtype='float32')

            layer1 = np.random.random((5, 5)).astype('float32')
            layer2 = np.random.random((5, 4)).astype('float32')

            bilinear = paddle.nn.Bilinear(
                in1_features=5, in2_features=4, out_features=1000
            )
            ret = bilinear(data1, data2)

            exe.run(main)
            ret_fetch = exe.run(
                feed={'X1': layer1, 'X2': layer2}, fetch_list=[ret]
            )
            self.assertEqual(ret_fetch[0].shape, (5, 1000))


class TestBilinearAPIDygraph(unittest.TestCase):
    def test_api(self):
        paddle.disable_static()
        layer1 = np.random.random((5, 5)).astype('float32')
        layer2 = np.random.random((5, 4)).astype('float32')
        bilinear = paddle.nn.Bilinear(
            in1_features=5, in2_features=4, out_features=1000
        )
        ret = bilinear(paddle.to_tensor(layer1), paddle.to_tensor(layer2))
        self.assertEqual(ret.shape, [5, 1000])


if __name__ == "__main__":
    unittest.main()
