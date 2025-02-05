# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved
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
        self.register_buffer("buffer", paddle.randn([5, 1]))

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
        paddle.seed(2024)
        self.net = TestNet()

    def pir_setup(self):
        with paddle.pir_utils.DygraphPirGuard():
            model = paddle.jit.to_static(
                self.net,
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
            paddle.jit.save(
                model,
                os.path.join(
                    self.temp_dir.name,
                    'test_inference_save_or_load_sep_model/pir/model',
                ),
                separate_parameters=True,
            )

    def old_sep_ir_setup(self):
        with paddle.pir_utils.DygraphOldIrGuard():
            model = paddle.jit.to_static(
                self.net,
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
            paddle.jit.save(
                model,
                os.path.join(
                    self.temp_dir.name,
                    'test_inference_save_or_load_sep_model/ir/model',
                ),
                separate_parameters=True,
            )

    def old_combine_ir_setup(self):
        with paddle.pir_utils.DygraphOldIrGuard():
            model = paddle.jit.to_static(
                self.net,
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
            paddle.jit.save(
                model,
                os.path.join(
                    self.temp_dir.name,
                    'test_inference_save_or_load_sep_model/ir_c/inference',
                ),
            )

    def tearDown(self):
        self.temp_dir.cleanup()

    def init_sep_predictor(self):
        config = Config(
            os.path.join(
                self.temp_dir.name,
                'test_inference_save_or_load_sep_model/ir/',
            )
        )
        config.enable_use_gpu(256, 0)
        predictor = create_predictor(config)
        return predictor

    def init_combine_predictor(self):
        config = Config(
            os.path.join(
                self.temp_dir.name,
                'test_inference_save_or_load_sep_model/ir_c/inference.pdmodel',
            ),
            os.path.join(
                self.temp_dir.name,
                'test_inference_save_or_load_sep_model/ir_c/inference.pdiparams',
            ),
        )
        config.enable_use_gpu(256, 0)
        predictor = create_predictor(config)
        return predictor

    def init_pir_predictor(self):
        config = Config(
            os.path.join(
                self.temp_dir.name,
                'test_inference_save_or_load_sep_model/pir/',
            ),
        )
        config.enable_use_gpu(256, 0)
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

    def get_inorder_output(self, predictor):
        [input0_tensor, input1_tensor] = self.get_inputs()

        inputs = [input0_tensor, input1_tensor]
        outputs = predictor.run(inputs)

        return outputs[0]

    def judge_model_file(self):
        base_dir = os.path.join(
            self.temp_dir.name,
            'test_inference_save_or_load_sep_model/',
        )
        ir_path = os.path.join(base_dir, "ir")
        ir_c_path = os.path.join(base_dir, "ir_c")
        pir_path = os.path.join(base_dir, "pir")

        def check_ir_folder(folder_path):
            files = os.listdir(folder_path)
            has_not_pdiparams = any(
                file.endswith("pdiparams") for file in files
            )
            has_model = "__model__" in files
            return (not has_not_pdiparams) and has_model

        def check_ir_c_folder(folder_path):
            files = os.listdir(folder_path)
            has_pdmodel = any(file.endswith(".pdmodel") for file in files)
            has_pdiparams = any(file.endswith(".pdiparams") for file in files)
            return has_pdmodel and has_pdiparams

        def check_pir_folder(folder_path):
            files = os.listdir(folder_path)
            has_not_pdiparams = any(
                file.endswith("pdiparams") for file in files
            )
            has_json = "__model__.json" in files
            return (not has_not_pdiparams) and has_json

        return (
            check_ir_folder(ir_path)
            and check_ir_c_folder(ir_c_path)
            and check_pir_folder(pir_path)
        )

    def test_output(self):
        self.old_sep_ir_setup()
        predictor = self.init_sep_predictor()
        output = self.get_inorder_output(predictor)

        self.pir_setup()
        pir_predictor = self.init_pir_predictor()
        pir_output = self.get_inorder_output(pir_predictor)

        self.old_combine_ir_setup()
        combine_predictor = self.init_combine_predictor()
        ir_combine_output = self.get_inorder_output(combine_predictor)

        # check model file structure
        self.assertTrue(self.judge_model_file(), msg="Model file is wrong")
        # check ir_sep and pir_sep output
        np.testing.assert_allclose(
            output.numpy().flatten(), pir_output.numpy().flatten()
        )
        # check ir_sep and ir_combine output
        np.testing.assert_allclose(
            output.numpy().flatten(), ir_combine_output.numpy().flatten()
        )


if __name__ == '__main__':
    unittest.main()
