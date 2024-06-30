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
import re
import shutil
import unittest

import numpy as np

import paddle
from paddle.base import core
from paddle.inference import Config, PrecisionType, create_predictor

# define the e4m3/e5m2 constants
E4M3_MAX_POS = 448.0
E5M2_MAX_POS = 57344.0


def get_cuda_version():
    result = os.popen("nvcc --version").read()
    regex = r'release (\S+),'
    match = re.search(regex, result)
    if match:
        num = str(match.group(1))
        integer, decimal = num.split('.')
        return int(integer) * 1000 + int(float(decimal) * 10)
    else:
        return -1


def check_fp8_support() -> bool:
    """Return if fp8 support is available"""
    gpu_arch = (
        paddle.device.cuda.get_device_capability()[0] * 10
        + paddle.device.cuda.get_device_capability()[1]
    )
    if gpu_arch >= 90:  # hopper and above
        return True
    # Device compute capability 8.9 or higher required for FP8 execution.
    if gpu_arch < 89:  # pre-ada
        return False
    if get_cuda_version() < 12010:
        return False
    return True


class FP16TestNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input1, input2):
        type = "float8_e4m3fn"
        output = paddle.linalg.fp8_fp8_half_gemm_fused(
            paddle.cast(input1, type),
            paddle.cast(input2, type),
            transpose_x=False,
            transpose_y=True,
            output_dtype="float16",
        )
        return paddle.cast(output, "float32")


@unittest.skipIf(
    not core.is_compiled_with_cuda() or not check_fp8_support(),
    "Fp8 matmul requires CUDA >= 12.1 on Ada arch or hopper arch",
)
class TestFP8FP16Gemm(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.test_model = FP16TestNet()
        self.model_path = "./tmp_fp16_model/"
        self.path_prefix = self.model_path + "model"
        paddle.jit.save(
            self.test_model,
            self.path_prefix,
            input_spec=[
                paddle.static.InputSpec(
                    shape=[16, 64], dtype='float32', name="input1"
                ),
                paddle.static.InputSpec(
                    shape=[32, 64], dtype='float32', name="input2"
                ),
            ],
        )
        self.x = np.ones([16, 64], np.float32)
        self.y = np.ones([32, 64], np.float32)

    def inference(self):
        # Config
        config = Config(self.path_prefix + ".pdmodel", "")
        config.enable_use_gpu(100, 0, PrecisionType.Float32)
        config.enable_new_executor()

        # predictor
        predictor = create_predictor(config)

        # inference
        input_names = predictor.get_input_names()

        input_tensor_0 = predictor.get_input_handle(input_names[0])
        input_tensor_0.reshape(self.x.shape)
        input_tensor_0.copy_from_cpu(self.x)

        input_tensor_1 = predictor.get_input_handle(input_names[1])
        input_tensor_1.reshape(self.y.shape)
        input_tensor_1.copy_from_cpu(self.y)

        # run
        predictor.run()

        results = []
        # get out data from output tensor
        output_names = predictor.get_output_names()
        for i, name in enumerate(output_names):
            output_tensor = predictor.get_output_handle(name)
            output_data = output_tensor.copy_to_cpu()
            results.append(output_data)

        return results[0]

    def test(self):
        paddle.device.set_device("gpu")
        fp8_out = self.inference()
        fp32_out = np.dot(self.x, np.transpose(self.y))
        np.testing.assert_allclose(fp8_out, fp32_out, rtol=1e-5, atol=1e-5)
        shutil.rmtree(self.model_path)


class BF16TestNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input1, input2):
        type = "float8_e4m3fn"
        output = paddle.linalg.fp8_fp8_half_gemm_fused(
            paddle.cast(input1, type),
            paddle.cast(input2, type),
            transpose_x=False,
            transpose_y=True,
            output_dtype="bfloat16",
        )
        return paddle.cast(output, "float32")


@unittest.skipIf(
    not core.is_compiled_with_cuda() or not check_fp8_support(),
    "Fp8 matmul requires CUDA >= 12.1 on Ada arch or hopper arch",
)
class TestFP8BF16Gemm(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.test_model = BF16TestNet()
        self.model_path = "./tmp_fp16_model/"
        self.path_prefix = self.model_path + "model"
        paddle.jit.save(
            self.test_model,
            self.path_prefix,
            input_spec=[
                paddle.static.InputSpec(
                    shape=[16, 64], dtype='float32', name="input1"
                ),
                paddle.static.InputSpec(
                    shape=[32, 64], dtype='float32', name="input2"
                ),
            ],
        )
        self.x = np.ones([16, 64], np.float32)
        self.y = np.ones([32, 64], np.float32)

    def inference(self):
        # Config
        config = Config(self.path_prefix + ".pdmodel", "")
        config.enable_use_gpu(100, 0, PrecisionType.Float32)
        config.enable_new_executor()

        # predictor
        predictor = create_predictor(config)

        # inference
        input_names = predictor.get_input_names()

        input_tensor_0 = predictor.get_input_handle(input_names[0])
        input_tensor_0.reshape(self.x.shape)
        input_tensor_0.copy_from_cpu(self.x)

        input_tensor_1 = predictor.get_input_handle(input_names[1])
        input_tensor_1.reshape(self.y.shape)
        input_tensor_1.copy_from_cpu(self.y)

        # run
        predictor.run()

        results = []
        # get out data from output tensor
        output_names = predictor.get_output_names()
        for i, name in enumerate(output_names):
            output_tensor = predictor.get_output_handle(name)
            output_data = output_tensor.copy_to_cpu()
            results.append(output_data)

        return results[0]

    def test(self):
        paddle.device.set_device("gpu")
        fp8_out = self.inference()
        fp32_out = np.dot(self.x, np.transpose(self.y))
        np.testing.assert_allclose(fp8_out, fp32_out, rtol=1e-2, atol=1e-2)
        shutil.rmtree(self.model_path)


if __name__ == "__main__":
    unittest.main()
