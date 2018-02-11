#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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

import paddle.v2.fluid.layers as layers
from paddle.v2.fluid.framework import Program, program_guard


class TestBook(unittest.TestCase):
    def test_detection_output(self):
        program = Program()
        with program_guard(program):
            pb = layers.data(
                name='prior_box',
                shape=[10, 4],
                append_batch_size=False,
                dtype='float32')
            pbv = layers.data(
                name='prior_box_var',
                shape=[10, 4],
                append_batch_size=False,
                dtype='float32')
            loc = layers.data(
                name='target_box',
                shape=[21, 4],
                append_batch_size=False,
                dtype='float32')
            scores = layers.data(
                name='scores',
                shape=[2, 21, 10],
                append_batch_size=False,
                dtype='float32')
            out = layers.detection_output(
                scores=scores, loc=loc, prior_box=pb, prior_box_var=pbv)
            self.assertIsNotNone(out)
        print(str(program))


if __name__ == '__main__':
    unittest.main()
