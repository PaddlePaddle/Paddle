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
import shutil
import tempfile
import unittest

import numpy as np

import paddle
from paddle import nn, static
from paddle.inference import Config, PrecisionType, create_predictor

paddle.enable_static()


class SimpleNet(nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2D(
            in_channels=4,
            out_channels=4,
            kernel_size=3,
            stride=2,
            padding=0,
        )
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2D(
            in_channels=4,
            out_channels=2,
            kernel_size=3,
            stride=2,
            padding=0,
        )
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2D(
            in_channels=2,
            out_channels=1,
            kernel_size=3,
            stride=2,
            padding=0,
        )
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(729, 10)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x


class TestTRTOptimizationLevel(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.temp_dir.name, 'optimization_level', '')
        self.model_prefix = self.path + 'infer_model'

    def tearDown(self):
        shutil.rmtree(self.path)

    def build_model(self):
        image = static.data(
            name='img', shape=[None, 4, 224, 224], dtype='float32'
        )
        predict = SimpleNet()(image)
        exe = paddle.static.Executor(self.place)
        exe.run(paddle.static.default_startup_program())
        paddle.static.save_inference_model(
            self.model_prefix, [image], [predict], exe
        )

    def init_predictor(self):
        config = Config(
            self.model_prefix + '.json', self.model_prefix + '.pdiparams'
        )
        config.enable_use_gpu(256, 0, PrecisionType.Half)
        config.enable_tensorrt_engine(
            workspace_size=1 << 30,
            max_batch_size=1,
            min_subgraph_size=3,
            precision_mode=PrecisionType.Half,
            use_static=False,
            use_calib_mode=False,
        )
        config.enable_memory_optim()
        config.exp_disable_tensorrt_dynamic_shape_ops(True)
        config.disable_glog_info()
        config.set_tensorrt_optimization_level(0)
        self.assertEqual(config.tensorrt_optimization_level(), 0)
        predictor = create_predictor(config)
        return predictor

    def infer(self, predictor, img):
        input_names = predictor.get_input_names()
        for i, name in enumerate(input_names):
            input_tensor = predictor.get_input_handle(name)
            input_tensor.reshape(img[i].shape)
            input_tensor.copy_from_cpu(img[i].copy())
        predictor.run()
        results = []
        output_names = predictor.get_output_names()
        for i, name in enumerate(output_names):
            output_tensor = predictor.get_output_handle(name)
            output_data = output_tensor.copy_to_cpu()
            results.append(output_data)
        return results

    def test_optimization_level(self):
        self.build_model()
        predictor = self.init_predictor()
        img = np.ones((1, 4, 224, 224), dtype=np.float32)
        results = self.infer(predictor, img=[img])


if __name__ == '__main__':
    unittest.main()
