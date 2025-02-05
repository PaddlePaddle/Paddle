# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import unittest

import numpy as np

import paddle
from paddle.base.core import AnalysisConfig, create_paddle_predictor
from paddle.jit import to_static


class OpenVINOBaseTest(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        paddle.device.set_device("cpu")
        self.batch_size = 1
        self.infer_threads = 1
        self.precision = "float32"
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        self.temp_dir = current_dir
        self.model_name = "model"
        self.model_dir = None
        self.paddle_config = None
        self.openvino_config = None
        self.input_names = None
        self.input_shape_map = None
        self.output_names = None
        self.input_data = []
        self.output_expected = []
        self.output_openvino = []
        self.precision_map = {
            "int8": AnalysisConfig.Int8,
            "float16": AnalysisConfig.Half,
            "float32": AnalysisConfig.Float32,
        }

    def to_static(self, model, input_spec):
        self.model_dir = os.path.join(self.temp_dir, self.model_name)
        net = to_static(
            model,
            input_spec=input_spec,
            full_graph=True,
        )
        paddle.jit.save(net, os.path.join(self.model_dir, 'inference'))

    def prepare_paddle_config(self):
        if self.paddle_config is not None:
            return
        self.paddle_config = AnalysisConfig(
            os.path.join(self.model_dir, 'inference.pdmodel'),
            os.path.join(self.model_dir, 'inference.pdiparams'),
        )
        self.paddle_config.disable_gpu()
        self.paddle_config.switch_ir_optim(False)

    def prepare_openvino_config(self):
        if self.openvino_config is not None:
            return
        self.openvino_config = AnalysisConfig(
            os.path.join(self.model_dir, 'inference.pdmodel'),
            os.path.join(self.model_dir, 'inference.pdiparams'),
        )
        self.openvino_config.disable_gpu()
        self.openvino_config.enable_openvino_engine(
            self.precision_map[self.precision]
        )
        self.openvino_config.set_cpu_math_library_num_threads(
            self.infer_threads
        )
        cache_dir = os.path.join(self.model_dir, '__cache__')
        self.openvino_config.set_optim_cache_dir(cache_dir)

    def prepare_input(self):
        if len(self.input_data) != len(self.input_names):
            for name in self.input_names:
                new_shape = [
                    self.batch_size if x == -1 else x
                    for x in self.input_shape_map[name]
                ]
                self.input_data.append(
                    np.random.random(new_shape).astype("float32")
                )

    def run_paddle(self):
        if self.paddle_config is None:
            self.prepare_paddle_config()
        self.paddle_predictor = create_paddle_predictor(self.paddle_config)
        self.input_names = self.paddle_predictor.get_input_names()
        self.input_shape_map = self.paddle_predictor.get_input_tensor_shape()

        self.prepare_input()

        for idx, name in enumerate(self.input_names):
            tensor = self.paddle_predictor.get_input_tensor(name)
            tensor.copy_from_cpu(self.input_data[idx])

        self.paddle_predictor.zero_copy_run()

        self.output_names = self.paddle_predictor.get_output_names()
        for name in self.output_names:
            self.output_expected.append(
                self.paddle_predictor.get_output_tensor(name).copy_to_cpu()
            )

    def run_openvino(self):
        self.prepare_openvino_config()
        self.openvino_predictor = create_paddle_predictor(self.openvino_config)

        for idx, name in enumerate(self.input_names):
            tensor = self.openvino_predictor.get_input_tensor(name)
            tensor.copy_from_cpu(self.input_data[idx])

        self.openvino_predictor.zero_copy_run()

        for name in self.output_names:
            self.output_openvino.append(
                self.paddle_predictor.get_output_tensor(name).copy_to_cpu()
            )

    def check_result(self, rtol=1e-3, atol=1e-3):
        self.run_paddle()
        self.run_openvino()

        for i in range(len(self.output_expected)):
            np.testing.assert_allclose(
                self.output_expected[i],
                self.output_openvino[i],
                rtol=rtol,
                atol=atol,
            )
