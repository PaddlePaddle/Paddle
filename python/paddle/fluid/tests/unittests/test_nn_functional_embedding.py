#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
import paddle.nn.functional as functional


class BadInputTestOnehotV2(unittest.TestCase):
    def test_error(self):
        with fluid.program_guard(fluid.Program()):

            def test_bad_x():
                label = fluid.layers.data(
                    name="label",
                    shape=[4],
                    append_batch_size=False,
                    dtype="float32")
                emb = functional.embedding(x=label, num_classes=4)

            self.assertRaises(TypeError, test_bad_x)
