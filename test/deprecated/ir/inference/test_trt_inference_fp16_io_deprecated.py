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


class TestEnableLowPrecisionIO:
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        net = alexnet(True)
        model = to_static(
            net,
            input_spec=[InputSpec(shape=[None, 3, 224, 224], name='x')],
            full_graph=True,
        )
        paddle.jit.save(
            model, os.path.join(self.temp_dir.name, 'alexnet/inference')
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def get_fp32_output(self):
        predictor = self.init_predictor(low_precision_io=False)

        inputs = [
            paddle.to_tensor(0.1 * np.ones([1, 3, 224, 224]).astype(np.float32))
        ]

        outputs = predictor.run(inputs)

        return outputs[0]

    def get_fp16_output(self):
        predictor = self.init_predictor(low_precision_io=True)

        inputs = [
            paddle.to_tensor(0.1 * np.ones([1, 3, 224, 224]).astype(np.float16))
        ]

        outputs = predictor.run(inputs)

        return outputs[0]

    def test_output(self):
        if paddle.is_compiled_with_cuda():
            fp32_output = self.get_fp32_output()
            fp16_output = self.get_fp16_output()

        # if os.name == 'posix':
        #     np.testing.assert_allclose(
        #         fp32_output.numpy().flatten(),
        #         fp16_output.numpy().flatten(),
        #     )


class TestEnableLowPrecisionIOWithGPU(
    TestEnableLowPrecisionIO, unittest.TestCase
):
    def init_predictor(self, low_precision_io: bool):
        config = Config(
            os.path.join(self.temp_dir.name, 'alexnet/inference.pdmodel'),
            os.path.join(self.temp_dir.name, 'alexnet/inference.pdiparams'),
        )
        config.enable_use_gpu(256, 0, PrecisionType.Half)
        config.enable_memory_optim()
        config.enable_low_precision_io(low_precision_io)
        config.disable_glog_info()
        predictor = create_predictor(config)
        return predictor


class TestEnableLowPrecisionIOWithTRTAllGraph(
    TestEnableLowPrecisionIO, unittest.TestCase
):
    def init_predictor(self, low_precision_io: bool):
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
        config.enable_tensorrt_memory_optim(True, 1)
        config.enable_tuned_tensorrt_dynamic_shape()
        config.enable_new_executor()
        config.enable_low_precision_io(low_precision_io)
        config.disable_glog_info()
        predictor = create_predictor(config)
        return predictor


class TestEnableLowPrecisionIOWithTRTSubGraph(
    TestEnableLowPrecisionIO, unittest.TestCase
):
    def init_predictor(self, low_precision_io: bool):
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
        config.enable_tensorrt_memory_optim(True, 1)
        config.enable_tuned_tensorrt_dynamic_shape()
        config.enable_new_executor()
        config.enable_low_precision_io(low_precision_io)
        config.exp_disable_tensorrt_ops(["flatten_contiguous_range"])
        config.disable_glog_info()
        predictor = create_predictor(config)
        return predictor


if __name__ == '__main__':
    unittest.main()
