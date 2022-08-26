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

import os
import paddle
import unittest
import tempfile
import numpy as np
from paddle.static import InputSpec
from paddle.fluid.framework import _enable_legacy_dygraph
from paddle.jit.layer import Layer
from paddle.fluid.dygraph.dygraph_to_static.program_translator import ProgramTranslator

paddle.seed(1)


class Net(paddle.nn.Layer):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = paddle.nn.Linear(4, 4)
        self.fc2 = paddle.nn.Linear(4, 4)
        self._bias = 0.4

    @paddle.jit.to_static(input_spec=[InputSpec([None, 4], dtype='float32')])
    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = paddle.nn.functional.relu(out)
        out = paddle.mean(out)
        return out

    @paddle.jit.to_static(input_spec=[InputSpec([None, 4], dtype='float32')])
    def infer(self, input):
        out = self.fc2(input)
        out = out + self._bias
        out = paddle.mean(out)
        return out


class TestMultiLoad(unittest.TestCase):

    def test_multi_load(self):
        self.temp_dir = tempfile.TemporaryDirectory()

        x = paddle.full([2, 4], 2)
        model = Net()
        program_translator = ProgramTranslator()
        program_translator.enable(False)
        forward_out1 = model.forward(x)
        infer_out1 = model.infer(x)
        program_translator.enable(True)

        model_path = os.path.join(self.temp_dir.name, 'multi_program')
        paddle.jit.save(model, model_path, combine_params=True)
        place = paddle.CPUPlace()
        if paddle.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
        jit_layer = Layer()
        jit_layer.load(model_path, place)
        forward_out2 = jit_layer.forward(x)
        infer_out2 = jit_layer.infer(x)
        np.testing.assert_allclose(forward_out1, forward_out2[0], rtol=1e-05)
        np.testing.assert_allclose(infer_out1, infer_out2[0], rtol=1e-05)

        self.temp_dir.cleanup()


if __name__ == '__main__':
    unittest.main()
