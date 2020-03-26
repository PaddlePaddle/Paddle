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

import os
import unittest

import numpy as np
import paddle.fluid as fluid

from paddle.fluid.dygraph.dygraph_to_static import ProgramTranslator
from paddle.fluid.dygraph.jit import dygraph_to_static_output

np.random.seed(2020)

place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda() else fluid.CPUPlace(
)


class SimpleFcLayer(fluid.dygraph.Layer):
    def __init__(self, fc_size):
        super(SimpleFcLayer, self).__init__()
        self._linear = fluid.dygraph.Linear(fc_size, fc_size)

    @dygraph_to_static_output
    def forward(self, x):
        x = fluid.dygraph.to_variable(x)
        y = self._linear(x)
        z = self._linear(y)
        out = fluid.layers.mean(z, name='mean')
        return out


class TestDyToStaticSaveInferenceModel(unittest.TestCase):
    def test_save_inference_model(self):
        fc_size = 20

        x = np.random.random((fc_size, fc_size)).astype('float32')
        layer = SimpleFcLayer(fc_size)

        program_translator = ProgramTranslator.get_instance()
        adam = fluid.optimizer.SGD(learning_rate=0.001)
        program_translator.set_optimizer(adam, 'mean.tmp_0')

        for i in range(5):
            out = layer(x)

        main_program = ProgramTranslator.get_instance().main_program
        expected_persistable_vars = set(
            [layer._linear.weight.name, layer._linear.bias.name])

        infer_model_dir = "./test_dy2stat_save_inference_model"
        ProgramTranslator.get_instance().save_inference_model(infer_model_dir)
        saved_var_names = set([
            filename for filename in os.listdir(infer_model_dir)
            if filename != '__model__'
        ])
        self.assertEqual(saved_var_names, expected_persistable_vars)


if __name__ == '__main__':
    unittest.main()
