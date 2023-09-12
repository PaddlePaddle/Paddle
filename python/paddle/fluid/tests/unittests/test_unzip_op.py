#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
import paddle.fluid.core as core


class TestUnzipOp(unittest.TestCase):
    def test_result(self):
        """
        For unzip op
        """
        paddle.enable_static()
        if core.is_compiled_with_cuda():
            place = fluid.CUDAPlace(0)
            x = fluid.data(name='X', shape=[1], dtype='float64')
            lod = fluid.data(name='lod', shape=[6], dtype='int64')
            len = 4
            output = paddle.incubate.operators.unzip(x, lod, len)

            input = [1.0, 2.0, 3.0, 1.0, 2.0, 4.0]
            lod = [0, 3, 3, 3, 4, 6]

            feed = {
                'X': np.array(input).astype("float64"),
                'lod': np.array(lod).astype("int64"),
            }

            exe = fluid.Executor(place=place)
            exe.run(fluid.default_startup_program())
            res = exe.run(feed=feed, fetch_list=[output])
            out = [[1.0, 2.0, 3.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0], 
                   [1.0, 0.0, 0.0, 0.0],
                   [2.0, 4.0, 0.0, 0.0]]
            out_np = np.array(out, dtype="float64")
            assert (res == out_np).all(), "output is not right"


if __name__ == '__main__':
    unittest.main()
