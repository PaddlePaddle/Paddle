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
from paddle.base import core
from paddle.inference import Config, create_predictor

# -------------------- SkipIrPassTestNet ----------------------
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


class SkipIrPassTestNet(paddle.nn.Layer):
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


class TestSkipIrPassApi(InferencePassTest):
    def setUp(self):
        paddle.disable_static()
        self.test_model = SkipIrPassTestNet()
        self.input_data = (np.ones([1, 3, 32, 32])).astype('float32')
        self.path_prefix = "inference_test_models/skip_ir_pass_test"
        x = paddle.to_tensor(self.input_data)
        y = self.test_model(x)
        print(y.shape)
        paddle.jit.save(
            self.test_model,
            self.path_prefix,
            input_spec=[
                paddle.static.InputSpec(shape=[1, 3, 32, 32], dtype='float32')
            ],
        )

    def inference_with_ir_pass(self):
        # Config
        config = Config(
            self.path_prefix + ".pdmodel", self.path_prefix + ".pdiparams"
        )
        config.enable_use_gpu(100, 0)
        config.enable_tensorrt_engine(
            workspace_size=1 << 30,
            max_batch_size=1,
            min_subgraph_size=3,
            precision_mode=paddle.inference.PrecisionType.Float32,
            use_static=True,
            use_calib_mode=False,
        )
        config.set_trt_dynamic_shape_info(
            {"x": [1, 3, 16, 16]},
            {"x": [1, 3, 64, 64]},
            {"x": [1, 3, 32, 32]},
        )
        config.exp_disable_tensorrt_ops(["elemenwise_add"])
        config.enable_save_optim_model()

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
        out = out.numpy().flatten()
        return out

    def inference_skip_ir_pass(self):
        config = Config(
            self.path_prefix + "/cache" + "_optimized.pdmodel",
            self.path_prefix + "/cache" + "_optimized.pdiparams",
        )
        config.enable_use_gpu(100, 0)
        config.skip_ir_pass(True)

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
        out = out.numpy().flatten()
        return out

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            out_with_ir_pass = self.inference_with_ir_pass()
            out_wihout_ir_pass = self.inference_skip_ir_pass()
            np.testing.assert_allclose(
                out_with_ir_pass, out_wihout_ir_pass, rtol=5e-5, atol=1e-2
            )


if __name__ == "__main__":
    unittest.main()
