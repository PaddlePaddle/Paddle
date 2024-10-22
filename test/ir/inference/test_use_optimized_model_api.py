# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from inference_pass_test import InferencePassTest

import paddle
from paddle.framework import core
from paddle.inference import Config, create_predictor

# -------------------------- TestNet --------------------------
#            x
#          /   \
#     conv2d    \                                  x
#       |        \        IR/Pass                /   \
#   batch_norm  conv2d    ——————>   tensorrt_engine  conv2d
#       |        /                               \   /
#     relu      /                            elemenwise_add
#         \    /                                   |
#     elemenwise_add                               y
#           |
#           y
# -------------------------------------------------------------


class TestNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = paddle.nn.Conv2D(3, 6, kernel_size=3, bias_attr=False)
        self.bn1 = paddle.nn.BatchNorm2D(6)
        self.relu = paddle.nn.ReLU()
        self.conv2 = paddle.nn.Conv2D(3, 6, kernel_size=3, bias_attr=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x2 = self.conv2(x)
        y = paddle.add(x1, x2)
        return y


class UseOptimizedModel(InferencePassTest):
    def setUp(self):
        paddle.disable_static()
        self.test_model = TestNet()
        self.input_data = (np.ones([1, 3, 32, 32])).astype('float32')
        self.path_prefix = "inference_test_models/use_optimized_model_test"
        self.cache_dir = "inference_test_models/cache"
        paddle.jit.save(
            self.test_model,
            self.path_prefix,
            input_spec=[
                paddle.static.InputSpec(shape=[1, 3, 32, 32], dtype='float32')
            ],
        )

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            out_origin_model = self.inference()
            out_optimized_model = self.inference()
            np.testing.assert_allclose(
                out_origin_model, out_optimized_model, rtol=1e-5, atol=1e-2
            )

    def inference(self):
        # Config
        config = Config(
            self.path_prefix + ".json", self.path_prefix + ".pdiparams"
        )
        config.enable_use_gpu(100, 0)
        config.enable_tensorrt_engine(
            workspace_size=1 << 30,
            max_batch_size=1,
            min_subgraph_size=1,
            precision_mode=paddle.inference.PrecisionType.Float32,
            use_static=True,
            use_calib_mode=False,
        )
        config.enable_tuned_tensorrt_dynamic_shape()
        config.exp_disable_tensorrt_ops(["elementwise_add"])
        config.set_optim_cache_dir(self.cache_dir)
        config.use_optimized_model(True)

        # predictor
        predictor = create_predictor(config)

        # inference
        input_tensor = predictor.get_input_handle(
            predictor.get_input_names()[0]
        )
        input_tensor.reshape(self.input_data.shape)
        input_tensor.copy_from_cpu(self.input_data.copy())
        predictor.run()
        output_tensor = predictor.get_output_handle(
            predictor.get_output_names()[0]
        )
        out = output_tensor.copy_to_cpu()
        out = np.array(out).flatten()
        return out


if __name__ == "__main__":
    unittest.main()
