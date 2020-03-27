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
import paddle
import paddle.fluid as fluid
from paddle.fluid.framework import Program, program_guard
import unittest


class TestClamp(unittest.TestCase):
    def test_clamp(self):
        program = Program()
        with program_guard(program):
            data_shape = [1, 3, 224, 224]
            images = fluid.data(name='image', shape=data_shape, dtype='float32')
            min = fluid.data(name='min', shape=[1], dtype='float32')
            max = fluid.data(name='max', shape=[1], dtype='float32')

            out1 = paddle.tensor.clamp(images, min=3., max=5.)
            out2 = paddle.tensor.clamp(images, min=min, max=max)
            self.assertEqual(out1.shape, images.shape)
            self.assertEqual(out2.shape, images.shape)


if __name__ == '__main__':
    unittest.main()
