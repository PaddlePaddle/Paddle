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

import unittest
import warnings

import paddle.v2.fluid as fluid
import paddle.v2.fluid.framework as framework
import paddle.v2.fluid.layers as layers
import paddle.v2.fluid.registry as registry


class TestRegistry(unittest.TestCase):
    def test_registry_layer(self):
        self.layer_type = "mean"
        program = framework.Program()

        x = fluid.layers.data(name='X', shape=[10, 10], dtype='float32')
        output = layers.mean(x)
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        X = np.random.random((10, 10)).astype("float32")
        mean_out = exe.run(program, feed={"X": X}, fetch_list=[output])
        self.assertAlmostEqual(np.mean(X), mean_out)
