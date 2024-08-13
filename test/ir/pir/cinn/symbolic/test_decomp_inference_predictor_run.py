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
        self.fc1 = paddle.nn.Linear(64, 32)
        self.fc2 = paddle.nn.Linear(64, 32)

    def forward(self, x1, x2):
        y1 = self.fc1(x1)
        y2 = self.fc2(x2)
        y3 = y1 + y2
        z = paddle.nn.functional.softmax(y3)
        return z


class TestPredictorRunWithTensor(unittest.TestCase):
    def setUp(self):
        self.use_gpu = paddle.is_compiled_with_cuda()
        np.random.seed(2023)
        self.shape = [4, 8, 16, 64]
        self.x = np.random.random(self.shape).astype(np.float32)
        self.y = np.random.random(self.shape).astype(np.float32)
        self.temp_dir = tempfile.TemporaryDirectory()
        net = TestNet()
        model = paddle.jit.to_static(
            net,
            input_spec=[
                paddle.static.InputSpec(
                    shape=[None, None, None, None],
                    dtype='float32',
                    name='input0',
                ),
                paddle.static.InputSpec(
                    shape=self.shape, dtype='float32', name='input1'
                ),
            ],
            full_graph=True,
        )
        paddle.jit.save(
            model,
            os.path.join(
                self.temp_dir.name, 'test_predictor_run_model/inference'
            ),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def init_predictor(self, use_pir: bool):
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
        if self.use_gpu:
            config.enable_use_gpu(256, 0)
        config.switch_ir_optim(False)
        config.enable_new_executor()
        if use_pir:
            config.enable_new_ir()
        predictor = create_predictor(config)
        return predictor

    def get_inputs(self):
        input0_tensor = paddle.to_tensor(self.x)
        input1_tensor = paddle.to_tensor(self.y)

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

    def test_output_prim_inorder(self):
        predictor = self.init_predictor(False)
        output = self.get_inorder_output(predictor)
        paddle.set_flags({'FLAGS_enable_pir_in_executor': True})
        paddle.core._set_prim_all_enabled(True)
        pir_predictor = self.init_predictor(True)
        pir_output = self.get_inorder_output(pir_predictor)
        paddle.core._set_prim_all_enabled(False)

        np.testing.assert_allclose(
            output.numpy().flatten(),
            pir_output.numpy().flatten(),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_output_prim_disorder(self):
        predictor = self.init_predictor(False)
        output = self.get_disorder_output(predictor)
        paddle.set_flags({'FLAGS_enable_pir_in_executor': True})
        paddle.core._set_prim_all_enabled(True)
        pir_predictor = self.init_predictor(True)
        pir_output = self.get_disorder_output(pir_predictor)
        paddle.core._set_prim_all_enabled(False)

        np.testing.assert_allclose(
            output.numpy().flatten(),
            pir_output.numpy().flatten(),
            rtol=1e-6,
            atol=1e-6,
        )


if __name__ == '__main__':
    unittest.main()
