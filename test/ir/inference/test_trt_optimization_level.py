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
from paddle.inference import Config, PrecisionType, create_predictor
from paddle.jit import to_static
from paddle.static import InputSpec
from paddle.vision.models import alexnet


class TestTRTOptimizationLevel(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def build_model(self):
        net = alexnet()
        model = to_static(
            net, input_spec=[InputSpec(shape=[None, 3, 224, 224], name='x')]
        )
        paddle.jit.save(
            model, os.path.join(self.temp_dir.name, 'alexnet/inference')
        )

    def init_predictor(self):
        config = Config(
            os.path.join(self.temp_dir.name, 'alexnet/inference.pdmodel'),
            os.path.join(self.temp_dir.name, 'alexnet/inference.pdiparams'),
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
        config.enable_tuned_tensorrt_dynamic_shape()
        config.enable_memory_optim()
        config.exp_disable_tensorrt_ops(["flatten_contiguous_range"])
        config.disable_glog_info()
        config.set_trt_optimization_level(0)
        self.assertEqual(config.tensorrt_optimization_level(), 0)
        predictor = create_predictor(config)
        return predictor

    def test_optimization_level(self):
        self.build_model()
        predictor = self.init_predictor()
        inputs = [
            paddle.to_tensor(0.1 * np.ones([1, 3, 224, 224]).astype(np.float32))
        ]
        outputs = predictor.run(inputs)


if __name__ == '__main__':
    unittest.main()
