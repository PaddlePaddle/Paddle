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
from paddle.inference import Config, create_predictor


class TestNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fc1 = paddle.nn.Linear(4, 4)
        self.fc2 = paddle.nn.Linear(4, 4)

    def forward(self, x1, x2):
        y1 = self.fc1(x1)
        y2 = self.fc2(x2)
        return y1 + y2


@unittest.skipIf(
    not paddle.is_compiled_with_cuda(), 'should compile with cuda.'
)
class TestPredictorRunWithTensor(unittest.TestCase):
    def setUp(self):

        self.temp_dir = tempfile.TemporaryDirectory()
        net = TestNet()
        model = paddle.jit.to_static(
            net,
            input_spec=[
                paddle.static.InputSpec(
                    shape=[None, 4], dtype='float32', name='input0'
                ),
                paddle.static.InputSpec(
                    shape=[None, 4], dtype='float32', name='input1'
                ),
            ],
            full_graph=True,
        )
        with paddle.pir_utils.OldIrGuard():
            paddle.jit.save(
                model,
                os.path.join(
                    self.temp_dir.name, 'test_predictor_run_model/inference'
                ),
            )

    def tearDown(self):
        self.temp_dir.cleanup()

    def init_predictor(self, use_pir: bool):
        with paddle.pir_utils.OldIrGuard():
            config = Config(
                os.path.join(
                    self.temp_dir.name,
                    'test_predictor_run_model/inference.pdmodel',
                ),
                os.path.join(
                    self.temp_dir.name,
                    'test_predictor_run_model/inference.pdiparams',
                ),
            )
            config.enable_use_gpu(256, 0)
            config.switch_ir_optim(False)
            # config.enable_memory_optim()
            config.enable_new_executor()
            if use_pir:
                config.enable_new_ir()
            predictor = create_predictor(config)
        return predictor

    def get_inputs(self):
        input0 = np.array([[1, 2, 3, 4], [2, 3, 4, 5]]).astype(np.float32)
        input1 = np.array([[0.1, 0.2, 0.3, 0.4], [1.2, 1.3, 1.4, 1.5]]).astype(
            np.float32
        )

        input0_tensor = paddle.to_tensor(input0)
        input1_tensor = paddle.to_tensor(input1)

        return [input0_tensor, input1_tensor]

    def get_disorder_output(self, predictor):
        [input0_tensor, input1_tensor] = self.get_inputs()

        input_names = predictor.get_input_names()
        input0_tensor.name = input_names[0]
        input1_tensor.name = input_names[1]

        # disorder
        inputs = [input1_tensor, input0_tensor]
        outputs = predictor.run(inputs)

        return outputs[0]

    def get_inorder_output(self, predictor):
        [input0_tensor, input1_tensor] = self.get_inputs()

        # inorder
        inputs = [input0_tensor, input1_tensor]
        outputs = predictor.run(inputs)

        return outputs[0]

    def test_output(self):
        predictor = self.init_predictor(False)
        output = self.get_inorder_output(predictor)
        pir_predictor = self.init_predictor(True)
        pir_output = self.get_disorder_output(pir_predictor)

        np.testing.assert_allclose(
            output.numpy().flatten(), pir_output.numpy().flatten()
        )


if __name__ == '__main__':
    unittest.main()
