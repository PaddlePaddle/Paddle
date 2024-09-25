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
from paddle import base
from paddle.base.core import AnalysisConfig, PassVersionChecker
from paddle.static import nn


class TensorRTSubgraphPassFcTest(InferencePassTest):
    def setUp(self):
        paddle.enable_static()
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(
                name="data", shape=[-1, 8], dtype="float32"
            )
            flatten_data = paddle.nn.Flatten()(data)
            fc_out = paddle.nn.Linear(flatten_data.shape[-1], 10)(flatten_data)
            reshape_out = paddle.reshape(x=fc_out, shape=[1, 10])
        self.feeds = {
            "data": np.random.random([1, 8]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassFcTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False
        )
        self.dynamic_shape_params = (
            TensorRTSubgraphPassFcTest.DynamicShapeParam(
                {'data': [1, 8]},
                {'data': [32, 8]},
                {'data': [1, 8]},
                False,
            )
        )
        self.fetch_list = [reshape_out]

    def test_check_output(self):
        if paddle.is_compiled_with_cuda():
            use_gpu = True
            # TRT output shape of fc is (1, 100, 1, 1). To compare the output value only, flatten the results.
            self.check_output_with_option(use_gpu, flatten=True)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass')
            )


class TensorRTSubgraphPassConcatTest(InferencePassTest):
    def setUp(self):
        with base.program_guard(self.main_program, self.startup_program):
            data1 = paddle.static.data(
                name="data1", shape=[-1, 3, 64, 64], dtype="float32"
            )
            data2 = paddle.static.data(
                name="data2", shape=[-1, 3, 64, 64], dtype="float32"
            )
            concat_out = paddle.concat([data1, data2], axis=2)
            out = nn.batch_norm(concat_out, is_test=True)
        self.feeds = {
            "data1": np.random.random([1, 3, 64, 64]).astype("float32"),
            "data2": np.random.random([1, 3, 64, 64]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassConcatTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False
        )
        self.dynamic_shape_params = (
            TensorRTSubgraphPassConcatTest.DynamicShapeParam(
                {'data1': [1, 3, 64, 64], 'data2': [1, 3, 64, 64]},
                {'data1': [32, 3, 64, 64], 'data2': [32, 3, 64, 64]},
                {'data1': [1, 3, 64, 64], 'data2': [1, 3, 64, 64]},
                False,
            )
        )
        self.fetch_list = [out]

    def test_check_output(self):
        if paddle.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass')
            )


class TensorRTSubgraphPassSplitTest(InferencePassTest):
    def setUp(self):
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(
                name="data", shape=[-1, 3, 64, 64], dtype="float32"
            )
            split_out = paddle.split(data, axis=-1, num_or_sections=2)
            out = nn.batch_norm(split_out[0], is_test=True)
        self.feeds = {
            "data": np.random.random([1, 3, 64, 64]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassSplitTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False
        )
        self.dynamic_shape_params = (
            TensorRTSubgraphPassSplitTest.DynamicShapeParam(
                {'data': [1, 3, 64, 64]},
                {'data': [32, 3, 64, 64]},
                {'data': [1, 3, 64, 64]},
                False,
            )
        )
        self.fetch_list = [out]

    def test_check_output(self):
        if paddle.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass')
            )


class TensorRTSubgraphPassSplitSerializeTest(InferencePassTest):
    def setUp(self):
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(
                name="data", shape=[-1, 3, 64, 64], dtype="float32"
            )
            split_out = paddle.split(data, axis=-1, num_or_sections=2)
            out = nn.batch_norm(split_out[0], is_test=True)
        self.feeds = {
            "data": np.random.random([1, 3, 64, 64]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = (
            TensorRTSubgraphPassSplitSerializeTest.TensorRTParam(
                1 << 30, 32, 0, AnalysisConfig.Precision.Float32, True, False
            )
        )
        self.dynamic_shape_params = (
            TensorRTSubgraphPassSplitSerializeTest.DynamicShapeParam(
                {'data': [1, 3, 64, 64]},
                {'data': [32, 3, 64, 64]},
                {'data': [1, 3, 64, 64]},
                False,
            )
        )
        self.fetch_list = [out]

    def test_check_output(self):
        if paddle.is_compiled_with_cuda():
            use_gpu = True
            opt_path = os.path.join(self.path, '_opt_cache')
            if os.path.exists(opt_path):
                shutil.rmtree(opt_path)
            self.check_output_with_option(use_gpu)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass')
            )


class TensorRTSubgraphPassDynamicSplitFp16SerializeTest(InferencePassTest):
    def setUp(self):
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(
                name="data", shape=[-1, 3, 64, 64], dtype="float32"
            )
            split_out = paddle.split(data, axis=-1, num_or_sections=2)
            out = nn.batch_norm(split_out[0], is_test=True)
        self.feeds = {
            "data": np.random.random([1, 3, 64, 64]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassSplitTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Half, True, False
        )
        self.dynamic_shape_params = (
            TensorRTSubgraphPassDynamicSplitFp16SerializeTest.DynamicShapeParam(
                {'data': [1, 3, 8, 64]},
                {'data': [1, 3, 512, 64]},
                {'data': [1, 3, 256, 64]},
                False,
            )
        )
        self.fetch_list = [out]

    def test_check_output(self):
        if paddle.is_compiled_with_cuda():
            use_gpu = True
            opt_path = os.path.join(self.path, '_opt_cache')
            if os.path.exists(opt_path):
                shutil.rmtree(opt_path)
            self.check_output_with_option(use_gpu, 1e-3)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass')
            )


class TensorRTSubgraphPassInstanceNormTest(InferencePassTest):
    def setUp(self):
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(
                name="data", shape=[-1, 3, 64, 64], dtype="float32"
            )
            param_attr = base.ParamAttr(
                name='instance_norm_w',
                initializer=paddle.nn.initializer.Constant(value=1.0),
            )
            bias_attr = base.ParamAttr(
                name='instance_norm_b',
                initializer=paddle.nn.initializer.Constant(value=0.0),
            )
            out = paddle.nn.InstanceNorm2D(
                num_features=3, weight_attr=param_attr, bias_attr=bias_attr
            )(data)
        self.feeds = {
            "data": np.random.random([1, 3, 64, 64]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = (
            TensorRTSubgraphPassInstanceNormTest.TensorRTParam(
                1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False
            )
        )
        self.dynamic_shape_params = (
            TensorRTSubgraphPassInstanceNormTest.DynamicShapeParam(
                {'data': [1, 3, 64, 64]},
                {'data': [32, 3, 64, 64]},
                {'data': [1, 3, 64, 64]},
                False,
            )
        )
        self.fetch_list = [out]

    def test_check_output(self):
        if paddle.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, atol=1e-4, flatten=True)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass')
            )


class TensorRTSubgraphPassTransposeTest(InferencePassTest):
    def setUp(self):
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(
                name="data", shape=[-1, 6, 64, 64], dtype="float32"
            )
            transpose_out = self.append_transpose(data)
            out = nn.batch_norm(transpose_out, is_test=True)
        self.feeds = {
            "data": np.random.random([1, 6, 64, 64]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassTransposeTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False
        )
        self.dynamic_shape_params = (
            TensorRTSubgraphPassTransposeTest.DynamicShapeParam(
                {'data': [1, 6, 64, 64]},
                {'data': [32, 6, 64, 64]},
                {'data': [1, 6, 64, 64]},
                False,
            )
        )
        self.fetch_list = [out]

    def append_transpose(self, data):
        return paddle.transpose(data, [0, 3, 1, 2])

    def test_check_output(self):
        if paddle.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass')
            )


class TensorRTSubgraphPassLayerNormTest(InferencePassTest):
    def setUp(self):
        self.set_params()
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(
                name="data", shape=[-1, 3, 64, 64], dtype="float32"
            )
            out = paddle.nn.LayerNorm(data.shape[self.begin_norm_axis :])(data)
        self.feeds = {
            "data": np.random.random([1, 3, 64, 64]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassLayerNormTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False
        )
        self.dynamic_shape_params = (
            TensorRTSubgraphPassLayerNormTest.DynamicShapeParam(
                {'data': [1, 3, 64, 64]},
                {'data': [32, 3, 64, 64]},
                {'data': [1, 3, 64, 64]},
                False,
            )
        )
        self.fetch_list = [out]

    def set_params(self):
        self.begin_norm_axis = 1

    def test_check_output(self):
        if paddle.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass')
            )


class TensorRTSubgraphPassLayerNormDynamicTest(InferencePassTest):
    def setUp(self):
        self.set_params()
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(
                name="data", shape=[-1, 3, 64, 64], dtype="float32"
            )
            out = paddle.nn.LayerNorm(data.shape[self.begin_norm_axis :])(data)
        self.feeds = {
            "data": np.random.random([1, 3, 64, 64]).astype("float32"),
        }
        self.set_trt_params()
        self.fetch_list = [out]

    def set_trt_params(self):
        self.enable_trt = True
        self.trt_parameters = (
            TensorRTSubgraphPassLayerNormDynamicTest.TensorRTParam(
                1 << 30, 32, 0, self.precision, self.serialize, False
            )
        )
        self.dynamic_shape_params = (
            TensorRTSubgraphPassLayerNormDynamicTest.DynamicShapeParam(
                {
                    'data': [1, 3, 64, 64],
                },
                {
                    'data': [8, 8, 64, 64],
                },
                {
                    'data': [4, 4, 64, 64],
                },
                False,
            )
        )

    def set_params(self):
        self.begin_norm_axis = 2
        self.precision = AnalysisConfig.Precision.Float32
        self.serialize = True

    def test_check_output(self):
        opt_path = os.path.join(self.path, '_opt_cache')
        if os.path.exists(opt_path):
            shutil.rmtree(opt_path)
        if paddle.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass')
            )


class TensorRTSubgraphPassLayerNormDynamicFP16Test(
    TensorRTSubgraphPassLayerNormDynamicTest
):
    def set_params(self):
        self.begin_norm_axis = 2
        self.precision = AnalysisConfig.Precision.Half
        self.serialize = True

    def test_check_output(self):
        opt_path = os.path.join(self.path, '_opt_cache')
        if os.path.exists(opt_path):
            shutil.rmtree(opt_path)
        if paddle.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, atol=0.01, rtol=0.01)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass')
            )


class TensorRTSubgraphPassLayerNormBeginNormAxis2Test(
    TensorRTSubgraphPassLayerNormTest
):
    def set_params(self):
        self.begin_norm_axis = 2


class TensorRTSubgraphPassLayerNormBeginNormAxis3Test(
    TensorRTSubgraphPassLayerNormTest
):
    def set_params(self):
        self.begin_norm_axis = 3


class TensorRTSubgraphPassElementwiseTest(InferencePassTest):
    def setUp(self):
        with base.program_guard(self.main_program, self.startup_program):
            data1 = paddle.static.data(
                name="data1", shape=[-1, 3, 64, 64], dtype="float32"
            )
            data2 = paddle.static.data(
                name="data2", shape=[-1, 3, 64, 64], dtype="float32"
            )
            eltwise_out = self.append_eltwise(data1, data2)
            out = nn.batch_norm(eltwise_out, is_test=True)
        self.feeds = {
            "data1": np.random.random([1, 3, 64, 64]).astype("float32"),
            "data2": np.random.random([1, 3, 64, 64]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassElementwiseTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False
        )
        self.dynamic_shape_params = (
            TensorRTSubgraphPassElementwiseTest.DynamicShapeParam(
                {'data1': [1, 3, 64, 64], 'data2': [1, 3, 64, 64]},
                {'data1': [32, 3, 64, 64], 'data2': [32, 3, 64, 64]},
                {'data1': [1, 3, 64, 64], 'data2': [1, 3, 64, 64]},
                False,
            )
        )
        self.fetch_list = [out]

    def append_eltwise(self, data1, data2):
        return paddle.add(x=data1, y=data2)

    def test_check_output(self):
        if paddle.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass')
            )


class TensorRTSubgraphPassElementwiseMulTest(
    TensorRTSubgraphPassElementwiseTest
):
    def append_eltwise(self, data1, data2):
        return paddle.multiply(x=data1, y=data2)


class TensorRTSubgraphPassElementwiseSerializeTest(
    TensorRTSubgraphPassElementwiseTest
):
    def setUp(self):
        super().setUp()
        self.trt_parameters = TensorRTSubgraphPassElementwiseTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Float32, True, False
        )

    def test_check_output(self):
        opt_path = os.path.join(self.path, '_opt_cache')
        if os.path.exists(opt_path):
            shutil.rmtree(opt_path)
        super().test_check_output()


class TensorRTSubgraphPassElementwiseBroadcastDynamicTest(InferencePassTest):
    def setUp(self):
        with base.program_guard(self.main_program, self.startup_program):
            data1 = paddle.static.data(
                name="data1", shape=[-1, 3, 64, 64], dtype="float32"
            )
            data2 = paddle.static.data(
                name="data2", shape=[64, 64], dtype="float32"
            )
            eltwise_out = self.append_eltwise(data1, data2)
            out = nn.batch_norm(eltwise_out, is_test=True)
        self.feeds = {
            "data1": np.random.random([1, 3, 64, 64]).astype("float32"),
            "data2": np.random.random([64, 64]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = (
            TensorRTSubgraphPassElementwiseBroadcastDynamicTest.TensorRTParam(
                1 << 30, 32, 0, AnalysisConfig.Precision.Float32, True, False
            )
        )
        self.dynamic_shape_params = TensorRTSubgraphPassElementwiseBroadcastDynamicTest.DynamicShapeParam(
            {'data1': [1, 3, 8, 64], 'data2': [8, 64]},
            {'data1': [1, 3, 512, 64], 'data2': [512, 64]},
            {'data1': [1, 3, 256, 64], 'data2': [256, 64]},
            False,
        )
        self.fetch_list = [out]

    def append_eltwise(self, data1, data2):
        return paddle.add(x=data1, y=data2)

    def test_check_output(self):
        opt_path = os.path.join(self.path, '_opt_cache')
        if os.path.exists(opt_path):
            shutil.rmtree(opt_path)
        if paddle.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass')
            )


if __name__ == "__main__":
    unittest.main()
