#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import sys

sys.path.append("..")

from op_test import OpTest
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid.op import Operator
import paddle


class TestSamplingIdShape(unittest.TestCase):

    def test_shape(self):
        paddle.enable_static()
        x = fluid.layers.data(name='x', shape=[3], dtype='float32')
        output = fluid.layers.sampling_id(x)

        place = fluid.XPUPlace(0)
        exe = fluid.Executor(place=place)
        exe.run(fluid.default_startup_program())

        feed = {
            'x': np.array([[0.2, 0.3, 0.5], [0.2, 0.3, 0.4]], dtype='float32')
        }
        output_np = exe.run(feed=feed, fetch_list=[output])[0]

        self.assertEqual(output.shape[0], -1)
        self.assertEqual(len(output.shape), 1)
        self.assertEqual(output_np.shape[0], 2)
        self.assertEqual(len(output_np.shape), 1)


if __name__ == "__main__":
    unittest.main()
