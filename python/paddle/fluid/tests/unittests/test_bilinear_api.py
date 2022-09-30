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
from op_test import OpTest

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import numpy as np


class TestBilinearAPI(unittest.TestCase):

    def test_api(self):
        with fluid.program_guard(fluid.default_startup_program(),
                                 fluid.default_main_program()):
            if core.is_compiled_with_cuda():
                place = core.CUDAPlace(0)
            else:
                place = core.CPUPlace()
            exe = fluid.Executor(place)

            data1 = fluid.data(name='X1', shape=[5, 5], dtype='float32')
            data2 = fluid.data(name='X2', shape=[5, 4], dtype='float32')

            layer1 = np.random.random((5, 5)).astype('float32')
            layer2 = np.random.random((5, 4)).astype('float32')

            bilinear = paddle.nn.Bilinear(in1_features=5,
                                          in2_features=4,
                                          out_features=1000)
            ret = bilinear(data1, data2)

            exe.run(fluid.default_startup_program())
            ret_fetch = exe.run(feed={
                'X1': layer1,
                'X2': layer2
            },
                                fetch_list=[ret.name])
            self.assertEqual(ret_fetch[0].shape, (5, 1000))


class TestBilinearAPIDygraph(unittest.TestCase):

    def test_api(self):
        paddle.disable_static()
        layer1 = np.random.random((5, 5)).astype('float32')
        layer2 = np.random.random((5, 4)).astype('float32')
        bilinear = paddle.nn.Bilinear(in1_features=5,
                                      in2_features=4,
                                      out_features=1000)
        ret = bilinear(paddle.to_tensor(layer1), paddle.to_tensor(layer2))
        self.assertEqual(ret.shape, [5, 1000])


if __name__ == "__main__":
    unittest.main()
