# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import unittest

import numpy as np
from inference_pass_test import InferencePassTest

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.nn.functional as F
import paddle.static.nn as nn
from paddle.fluid.core import AnalysisConfig, PassVersionChecker


class TensorRTSubgraphPassActivationTest(InferencePassTest):
    def setUpTensorRTParam(self):
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassActivationTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False
        )

    def setUp(self):
        self.setUpTensorRTParam()
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 6, 32, 32], dtype="float32"
            )
            act_out = self.append_act(data)
            out = nn.batch_norm(act_out, is_test=True)
        self.feeds = {
            "data": np.random.random([1, 6, 32, 32]).astype("float32"),
        }
        self.fetch_list = [out]

    def append_act(self, x):
        return F.relu(x)

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            if os.path.exists(self.path + "_opt_cache"):
                shutil.rmtree(self.path + "_opt_cache")
            if (
                self.trt_parameters.precision
                == AnalysisConfig.Precision.Float32
            ):
                self.check_output_with_option(use_gpu)
            else:
                self.check_output_with_option(use_gpu, 1e-3)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass')
            )


class TensorRTSubgraphPassLeakyReluTest(TensorRTSubgraphPassActivationTest):
    def append_act(self, x):
        return paddle.nn.functional.leaky_relu(x)


class TensorRTSubgraphPassRelu6Test(TensorRTSubgraphPassActivationTest):
    def append_act(self, x):
        return paddle.nn.functional.relu6(x)


class TensorRTSubgraphPassSoftMaxTest(TensorRTSubgraphPassActivationTest):
    def append_act(self, x):
        return paddle.nn.functional.softmax(x)


class TensorRTSubgraphPassSigmoidTest(TensorRTSubgraphPassActivationTest):
    def append_act(self, x):
        return paddle.nn.functional.sigmoid(x)


class TensorRTSubgraphPassHardSwishTest(TensorRTSubgraphPassActivationTest):
    def append_act(self, x):
        return paddle.nn.functional.hardswish(x)


class TensorRTSubgraphPassHardSigmoidTest(TensorRTSubgraphPassActivationTest):
    def append_act(self, x):
        return paddle.nn.functional.hardsigmoid(x)


class TensorRTSubgraphPassHardSwishPluginTest(
    TensorRTSubgraphPassActivationTest
):
    def append_act(self, x):
        return paddle.nn.functional.hardswish(x)


class TensorRTSubgraphPassClipTest(TensorRTSubgraphPassActivationTest):
    def append_act(self, x):
        return fluid.layers.clip(x, 0, 1)


class TensorRTSubgraphPassTanhTest(TensorRTSubgraphPassActivationTest):
    def append_act(self, x):
        return paddle.tanh(x)


class TensorRTSubgraphPassSwishTest(TensorRTSubgraphPassActivationTest):
    def setUpTensorRTParam(self):
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassActivationTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Float32, True, False
        )

    def append_act(self, x):
        return paddle.nn.functional.swish(x)


class TensorRTSubgraphPassSwishFp16SerializeTest(
    TensorRTSubgraphPassActivationTest
):
    def setUpTensorRTParam(self):
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassActivationTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Half, True, False
        )

    def append_act(self, x):
        return paddle.nn.functional.swish(x)


class TensorRTSubgraphPassDynamicSwishFp16SerializeTest(
    TensorRTSubgraphPassActivationTest
):
    def setUpTensorRTParam(self):
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassActivationTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Half, True, False
        )
        self.dynamic_shape_params = (
            TensorRTSubgraphPassActivationTest.DynamicShapeParam(
                {'data': [1, 6, 8, 8]},
                {'data': [1, 6, 128, 128]},
                {'data': [1, 6, 64, 64]},
                False,
            )
        )

    def append_act(self, x):
        return paddle.nn.functional.swish(x)


class TensorRTSubgraphPassMishTest(TensorRTSubgraphPassActivationTest):
    def setUpTensorRTParam(self):
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassActivationTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Float32, True, False
        )

    def append_act(self, x):
        return paddle.nn.functional.mish(x)


class TensorRTSubgraphPassMishFp16SerializeTest(
    TensorRTSubgraphPassActivationTest
):
    def setUpTensorRTParam(self):
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassActivationTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Half, True, False
        )

    def append_act(self, x):
        return paddle.nn.functional.mish(x)


class TensorRTSubgraphPassDynamicMishFp16SerializeTest(
    TensorRTSubgraphPassActivationTest
):
    def setUpTensorRTParam(self):
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassActivationTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Half, False, False
        )
        self.dynamic_shape_params = (
            TensorRTSubgraphPassActivationTest.DynamicShapeParam(
                {'data': [1, 6, 8, 8]},
                {'data': [1, 6, 128, 128]},
                {'data': [1, 6, 64, 64]},
                False,
            )
        )

    def append_act(self, x):
        return paddle.nn.functional.mish(x)


class TensorRTSubgraphPassPreluAllTest(TensorRTSubgraphPassActivationTest):
    def append_act(self, x):
        return paddle.static.nn.prelu(x, mode='all')


class TensorRTSubgraphPassPreluChannelTest(TensorRTSubgraphPassActivationTest):
    def append_act(self, x):
        return paddle.static.nn.prelu(x, mode='channel')


class TensorRTSubgraphPassPreluElementTest(TensorRTSubgraphPassActivationTest):
    def append_act(self, x):
        return paddle.static.nn.prelu(x, mode='element')


class TensorRTSubgraphPassPreluDynamicTest(TensorRTSubgraphPassActivationTest):
    def setUpTensorRTParam(self):
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassActivationTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False
        )
        self.dynamic_shape_params = (
            TensorRTSubgraphPassActivationTest.DynamicShapeParam(
                {'data': [1, 6, 8, 8]},
                {'data': [1, 6, 128, 128]},
                {'data': [1, 6, 64, 64]},
                False,
            )
        )

    def append_act(self, x):
        return paddle.static.nn.prelu(x, mode='all')


class TensorRTSubgraphPassPreluFp16Test(TensorRTSubgraphPassActivationTest):
    def setUpTensorRTParam(self):
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassActivationTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Half, False, False
        )

    def append_act(self, x):
        return paddle.static.nn.prelu(x, mode='all')


class TensorRTSubgraphPassPreluFp16SerializeTest(
    TensorRTSubgraphPassActivationTest
):
    def setUpTensorRTParam(self):
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassActivationTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Half, True, False
        )

    def append_act(self, x):
        return paddle.static.nn.prelu(x, mode='all')


class TensorRTSubgraphPassPreluFp16DynamicTest(
    TensorRTSubgraphPassActivationTest
):
    def setUpTensorRTParam(self):
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassActivationTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Half, False, False
        )
        self.dynamic_shape_params = (
            TensorRTSubgraphPassActivationTest.DynamicShapeParam(
                {'data': [1, 6, 8, 8]},
                {'data': [1, 6, 128, 128]},
                {'data': [1, 6, 64, 64]},
                False,
            )
        )

    def append_act(self, x):
        return paddle.static.nn.prelu(x, mode='all')


class TensorRTSubgraphPassPreluFp16DynamicSerializeTest(
    TensorRTSubgraphPassActivationTest
):
    def setUpTensorRTParam(self):
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassActivationTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Half, True, False
        )
        self.dynamic_shape_params = (
            TensorRTSubgraphPassActivationTest.DynamicShapeParam(
                {'data': [1, 6, 8, 8]},
                {'data': [1, 6, 128, 128]},
                {'data': [1, 6, 64, 64]},
                False,
            )
        )

    def append_act(self, x):
        return paddle.static.nn.prelu(x, mode='all')


class TensorRTSubgraphPassGeluTest(TensorRTSubgraphPassActivationTest):
    def append_act(self, x):
        return paddle.nn.functional.gelu(x)


class TensorRTSubgraphPassGeluDynamicTest(TensorRTSubgraphPassActivationTest):
    def setUpTensorRTParam(self):
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassActivationTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False
        )
        self.dynamic_shape_params = (
            TensorRTSubgraphPassActivationTest.DynamicShapeParam(
                {'data': [1, 6, 8, 8]},
                {'data': [1, 6, 128, 128]},
                {'data': [1, 6, 64, 64]},
                False,
            )
        )

    def append_act(self, x):
        return paddle.nn.functional.gelu(x)


class TensorRTSubgraphPassGeluFp16Test(TensorRTSubgraphPassActivationTest):
    def setUpTensorRTParam(self):
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassActivationTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Half, False, False
        )

    def append_act(self, x):
        return paddle.nn.functional.gelu(x)


class TensorRTSubgraphPassGeluFp16SerializeTest(
    TensorRTSubgraphPassActivationTest
):
    def setUpTensorRTParam(self):
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassActivationTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Half, True, False
        )

    def append_act(self, x):
        return paddle.nn.functional.gelu(x)


class TensorRTSubgraphPassGeluFp16DynamicTest(
    TensorRTSubgraphPassActivationTest
):
    def setUpTensorRTParam(self):
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassActivationTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Half, False, False
        )
        self.dynamic_shape_params = (
            TensorRTSubgraphPassActivationTest.DynamicShapeParam(
                {'data': [1, 6, 8, 8]},
                {'data': [1, 6, 128, 128]},
                {'data': [1, 6, 64, 64]},
                False,
            )
        )

    def append_act(self, x):
        return paddle.nn.functional.gelu(x)


class TensorRTSubgraphPassGeluFp16DynamicSerializeTest(
    TensorRTSubgraphPassActivationTest
):
    def setUpTensorRTParam(self):
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassActivationTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Half, True, False
        )
        self.dynamic_shape_params = (
            TensorRTSubgraphPassActivationTest.DynamicShapeParam(
                {'data': [1, 6, 8, 8]},
                {'data': [1, 6, 128, 128]},
                {'data': [1, 6, 64, 64]},
                False,
            )
        )

    def append_act(self, x):
        return paddle.nn.functional.gelu(x)


if __name__ == "__main__":
    unittest.main()
