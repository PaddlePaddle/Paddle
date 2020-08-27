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

import unittest
import numpy as np
import paddle.fluid as fluid
import paddle.nn.functional as functional


class EmbeddingUT(unittest.TestCase):
    def test_1(self):
        prog = fluid.Program()
        with fluid.program_guard(prog):

            def test_bad_x():
                initializer = fluid.initializer.NumpyArrayInitializer(
                    np.random.random(size=(128, 100)))

                param_attr = fluid.ParamAttr(
                    name="emb_weight",
                    learning_rate=0.5,
                    initializer=initializer,
                    trainable=True)

                weight = prog.global_block().create_parameter(
                    (128, 100), attr=param_attr, dtype="float32")

                label = fluid.layers.data(
                    name="label",
                    shape=[4],
                    append_batch_size=False,
                    dtype="int64")

                emb = functional.embedding(
                    input=label,
                    weight=weight,
                    is_sparse=True,
                    name="embedding")

            test_bad_x()


if __name__ == '__main__':
    unittest.main()
