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

from __future__ import print_function
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import numpy as np
import os
import shutil
import unittest


class TestPipeline(unittest.TestCase):
    """  TestCases for Pipeline Training. """

    def test_pipeline(self):
        with fluid.device_guard("cpu"):
            x = fluid.layers.data(
                name='x', shape=[1], dtype='int64', lod_level=0)
            y = fluid.layers.data(
                name='y', shape=[1], dtype='int64', lod_level=0)
            emb_x = layers.embedding(
                input=x,
                param_attr=fluid.ParamAttr(name="embx"),
                size=[10, 2],
                is_sparse=False)
            emb_y = layers.embedding(
                input=y,
                param_attr=fluid.ParamAttr(
                    name="emby", learning_rate=0.9),
                size=[10, 2],
                is_sparse=False)

        with fluid.device_guard("gpu:0"):
            concat = layers.concat([emb_x, emb_y], axis=1)

            fc = layers.fc(input=concat,
                           name="fc",
                           size=1,
                           num_flatten_dims=1,
                           bias_attr=False)
            loss = layers.reduce_mean(fc)

        optimizer = fluid.optimizer.SGD(learning_rate=0.5)
        optimizer = fluid.optimizer.PipelineOptimizer(
            optimizer, num_microbatches=2)
        optimizer.minimize(loss)


if __name__ == '__main__':
    unittest.main()
