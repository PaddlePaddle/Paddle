# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved
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
import tempfile
import unittest

import numpy as np

import paddle
from paddle.inference import Config, DataType, create_predictor

paddle.set_default_dtype('float64')


class TestNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fc1 = paddle.nn.Linear(4, 4)
        self.fc2 = paddle.nn.Linear(4, 4)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = paddle.nn.functional.relu(out)
        return out


@unittest.skipIf(
    not paddle.is_compiled_with_cuda(), 'should compile with cuda.'
)
class TestDoubleOnGPU(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        net = TestNet()
        model = paddle.jit.to_static(
            net,
            input_spec=[
                paddle.static.InputSpec(shape=[None, 4], dtype='float64')
            ],
            full_graph=True,
        )
        paddle.jit.save(
            model,
            os.path.join(
                self.temp_dir.name, 'test_inference_datatype_model/inference'
            ),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def init_predictor(self):
        config = Config(
            os.path.join(
                self.temp_dir.name,
                'test_inference_datatype_model/inference.json',
            ),
            os.path.join(
                self.temp_dir.name,
                'test_inference_datatype_model/inference.pdiparams',
            ),
        )
        config.enable_use_gpu(256, 0)
        config.enable_memory_optim()
        config.enable_new_executor()
        config.enable_new_ir()
        # NOTE(liuyuanle): Because double computing is not supported in our pass implementation,
        # we need to turn off IR optimization.
        config.switch_ir_optim(False)
        predictor = create_predictor(config)
        return predictor

    def test_output(self):
        predictor = self.init_predictor()

        input = np.ones((3, 4)).astype(np.float64)

        input_names = predictor.get_input_names()
        input_tensor = predictor.get_input_handle(input_names[0])
        input_tensor.reshape(input.shape)
        input_tensor.copy_from_cpu(input.copy())
        assert input_tensor.type() == DataType.FLOAT64

        predictor.run()

        output_names = predictor.get_output_names()
        output_tensor = predictor.get_output_handle(output_names[0])
        assert output_tensor.type() == DataType.FLOAT64

        output_data = output_tensor.copy_to_cpu()


if __name__ == '__main__':
    unittest.main()
