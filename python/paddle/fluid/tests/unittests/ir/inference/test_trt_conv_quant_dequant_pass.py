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

import unittest

import numpy as np
from quant_dequant_test import QuantDequantTest

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.nn.functional as F
from paddle.fluid.core import AnalysisConfig, PassVersionChecker


class QuantDequantTensorRTSubgraphPassConvTest(QuantDequantTest):
    def setUp(self):
        self.set_params()

        def network():
            self.data = fluid.data(
                name='data', shape=[1, 28, 28], dtype='float32'
            )
            data_reshape = paddle.reshape(self.data, shape=[1, 4, 14, 14])
            self.label = fluid.data(name='label', shape=[1, 1], dtype='int64')
            label_shape = paddle.reshape(self.label, shape=[1, 1, 1])
            conv_out = fluid.layers.conv2d(
                input=data_reshape,
                num_filters=self.conv_num_filters,
                filter_size=self.conv_filter_size,
                groups=self.conv_groups,
                padding=self.conv_padding,
                bias_attr=False,
                use_cudnn=self.use_cudnn,
                act=None,
            )
            if self.conv_padding == [1, 1]:
                cout = paddle.reshape(conv_out, shape=[1, 1, 10816])
            elif self.conv_padding == 'VALID':
                cout = paddle.reshape(conv_out, shape=[1, 1, 7744])
            elif self.conv_padding == 'SAME':
                cout = paddle.reshape(conv_out, shape=[1, 1, 12544])
            elif self.conv_groups == 4:
                cout = paddle.reshape(conv_out, shape=[1, 1, 10816])
            result = F.relu(cout)
            loss = paddle.nn.functional.cross_entropy(
                input=result,
                label=label_shape,
                reduction='none',
                use_softmax=False,
            )
            avg_loss = paddle.mean(loss)
            return avg_loss, result

        self.main_program.random_seed = 2
        self.startup_program.random_seed = 2
        self.test_main_program.random_seed = 2
        # self.test_startup_program.random_seed = 2
        with fluid.unique_name.guard():
            with fluid.program_guard(self.main_program, self.startup_program):
                self.loss, result = network()
                opt = fluid.optimizer.Adam(learning_rate=0.0001)
                opt.minimize(self.loss)
        with fluid.unique_name.guard():
            with fluid.program_guard(
                self.test_main_program, self.startup_program
            ):
                network()
        self.feeds = {"data": np.random.random([1, 28, 28]).astype("float32")}
        self.fetch_list = [result]
        self.enable_trt = True
        self.trt_parameters = (
            QuantDequantTensorRTSubgraphPassConvTest.TensorRTParam(
                1 << 30, 32, 0, AnalysisConfig.Precision.Int8, False, False
            )
        )
        self.activation_quantize_type = 'moving_average_abs_max'
        self.weight_quantize_type = 'channel_wise_abs_max'

    def set_params(self):
        self.conv_num_filters = 64
        self.conv_filter_size = 4
        self.conv_groups = 1
        self.conv_padding = [1, 1]
        self.use_cudnn = True

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(
                use_gpu, atol=1e-1, flatten=False, rtol=1e-1
            )
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass')
            )


class QuantDequantTensorRTSubgraphPassConvValidPaddingTest(
    QuantDequantTensorRTSubgraphPassConvTest
):
    def set_params(self):
        self.conv_num_filters = 64
        self.conv_filter_size = 4
        self.conv_groups = 1
        self.conv_padding = 'VALID'
        self.use_cudnn = True


class QuantDequantTensorRTSubgraphPassConvSamePaddingTest(
    QuantDequantTensorRTSubgraphPassConvTest
):
    def set_params(self):
        self.conv_num_filters = 64
        self.conv_filter_size = 4
        self.conv_groups = 1
        self.conv_padding = 'SAME'
        self.use_cudnn = True


class QuantDequantTensorRTSubgraphPassDWConvTest(
    QuantDequantTensorRTSubgraphPassConvTest
):
    def set_params(self):
        self.conv_num_filters = 64
        self.conv_filter_size = 4
        self.conv_groups = 4
        self.conv_padding = [1, 1]
        self.use_cudnn = True


class DynamicShapeQuantDequantTensorRTSubgraphPassConvTest(QuantDequantTest):
    def setUp(self):
        self.set_params()

        def network():
            self.data = fluid.data(
                name='data', shape=[1, 28, 28], dtype='float32'
            )
            data_reshape = paddle.reshape(self.data, shape=[1, 4, 14, 14])
            self.label = fluid.data(name='label', shape=[1, 1], dtype='int64')
            label_shape = paddle.reshape(self.label, shape=[1, 1, 1])
            conv_out = fluid.layers.conv2d(
                input=data_reshape,
                num_filters=self.conv_num_filters,
                filter_size=self.conv_filter_size,
                groups=self.conv_groups,
                padding=self.conv_padding,
                bias_attr=False,
                use_cudnn=self.use_cudnn,
                act=None,
            )
            cout = paddle.reshape(conv_out, shape=[1, 1, 10816])
            result = F.relu(cout)
            loss = paddle.nn.functional.cross_entropy(
                input=result,
                label=label_shape,
                reduction='none',
                use_softmax=False,
            )
            avg_loss = paddle.mean(loss)
            return avg_loss, result

        self.main_program.random_seed = 2
        self.startup_program.random_seed = 2
        self.test_main_program.random_seed = 2
        # self.test_startup_program.random_seed = 2
        with fluid.unique_name.guard():
            with fluid.program_guard(self.main_program, self.startup_program):
                self.loss, result = network()
                opt = fluid.optimizer.Adam(learning_rate=0.0001)
                opt.minimize(self.loss)
        with fluid.unique_name.guard():
            with fluid.program_guard(
                self.test_main_program, self.startup_program
            ):
                network()
        self.feeds = {"data": np.random.random([1, 28, 28]).astype("float32")}
        self.fetch_list = [result]
        self.enable_trt = True
        self.trt_parameters = (
            DynamicShapeQuantDequantTensorRTSubgraphPassConvTest.TensorRTParam(
                1 << 30, 32, 0, AnalysisConfig.Precision.Int8, False, False
            )
        )
        self.dynamic_shape_params = DynamicShapeQuantDequantTensorRTSubgraphPassConvTest.DynamicShapeParam(
            {
                "conv2d_0.tmp_0": [1, 4, 14, 14],
                "data": [1, 28, 28],
                "depthwise_conv2d_0.tmp_0": [1, 4, 14, 14],
                "reshape2_0.tmp_0": [1, 4, 14, 14],
                "reshape2_2.tmp_0": [1, 1, 10816],
            },
            {
                "conv2d_0.tmp_0": [4, 4, 14, 14],
                "data": [4, 28, 28],
                "depthwise_conv2d_0.tmp_0": [4, 4, 14, 14],
                "reshape2_0.tmp_0": [4, 4, 14, 14],
                "reshape2_2.tmp_0": [1, 1, 43264],
            },
            {
                "conv2d_0.tmp_0": [1, 4, 14, 14],
                "data": [1, 28, 28],
                "depthwise_conv2d_0.tmp_0": [1, 4, 14, 14],
                "reshape2_0.tmp_0": [1, 4, 14, 14],
                "reshape2_2.tmp_0": [1, 1, 10816],
            },
            False,
        )
        self.activation_quantize_type = 'moving_average_abs_max'
        self.weight_quantize_type = 'channel_wise_abs_max'

    def set_params(self):
        self.conv_num_filters = 64
        self.conv_filter_size = 4
        self.conv_groups = 1
        self.conv_padding = [1, 1]
        self.use_cudnn = True

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(
                use_gpu, atol=1e-1, flatten=False, rtol=1e-1
            )
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass')
            )


class QuantDequantTensorRTSubgraphPassConvTransposeTest(QuantDequantTest):
    def setUp(self):
        self.set_params()

        def network():
            self.data = fluid.data(
                name='data', shape=[1, 28, 28], dtype='float32'
            )
            data_reshape = paddle.reshape(self.data, shape=[1, 4, 14, 14])
            self.label = fluid.data(name='label', shape=[1, 1], dtype='int64')
            label_shape = paddle.reshape(self.label, shape=[1, 1, 1])
            conv_out = paddle.static.nn.conv2d_transpose(
                input=data_reshape,
                num_filters=self.conv_num_filters,
                filter_size=self.conv_filter_size,
                groups=self.conv_groups,
                padding=self.conv_padding,
                bias_attr=False,
                use_cudnn=self.use_cudnn,
                act=None,
            )
            if self.conv_padding == [1, 1]:
                cout = paddle.reshape(conv_out, shape=[1, 1, 14400])
            elif self.conv_padding == 'VALID':
                cout = paddle.reshape(conv_out, shape=[1, 1, 18496])
            elif self.conv_padding == 'SAME':
                cout = paddle.reshape(conv_out, shape=[1, 1, 12544])
            elif self.conv_groups == 4:
                cout = paddle.reshape(conv_out, shape=[1, 1, 10816])
            result = F.relu(cout)
            loss = paddle.nn.functional.cross_entropy(
                input=result,
                label=label_shape,
                reduction='none',
                use_softmax=False,
            )
            avg_loss = paddle.mean(loss)
            return avg_loss, result

        self.main_program.random_seed = 2
        self.startup_program.random_seed = 2
        self.test_main_program.random_seed = 2
        # self.test_startup_program.random_seed = 2
        with fluid.unique_name.guard():
            with fluid.program_guard(self.main_program, self.startup_program):
                self.loss, result = network()
                opt = fluid.optimizer.Adam(learning_rate=0.0001)
                opt.minimize(self.loss)
        with fluid.unique_name.guard():
            with fluid.program_guard(
                self.test_main_program, self.startup_program
            ):
                network()
        self.feeds = {"data": np.random.random([1, 28, 28]).astype("float32")}
        self.fetch_list = [result]
        self.enable_trt = True
        self.trt_parameters = (
            QuantDequantTensorRTSubgraphPassConvTransposeTest.TensorRTParam(
                1 << 30, 32, 0, AnalysisConfig.Precision.Int8, False, False
            )
        )
        self.activation_quantize_type = 'moving_average_abs_max'
        self.weight_quantize_type = 'channel_wise_abs_max'

    def set_params(self):
        self.conv_num_filters = 64
        self.conv_filter_size = 4
        self.conv_groups = 1
        self.conv_padding = [1, 1]
        self.use_cudnn = True

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(
                use_gpu, atol=1e-1, flatten=False, rtol=1e-1
            )
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass')
            )


class QuantDequantTensorRTSubgraphPassConvTransValidPaddingTest(
    QuantDequantTensorRTSubgraphPassConvTransposeTest
):
    def set_params(self):
        self.conv_num_filters = 64
        self.conv_filter_size = 4
        self.conv_groups = 1
        self.conv_padding = 'VALID'
        self.use_cudnn = True


class QuantDequantTensorRTSubgraphPassConvTransSamePaddingTest(
    QuantDequantTensorRTSubgraphPassConvTransposeTest
):
    def set_params(self):
        self.conv_num_filters = 64
        self.conv_filter_size = 4
        self.conv_groups = 1
        self.conv_padding = 'SAME'
        self.use_cudnn = True


class QuantDequantTensorRTSubgraphPassTransDWConvTest(
    QuantDequantTensorRTSubgraphPassConvTransposeTest
):
    def set_params(self):
        self.conv_num_filters = 64
        self.conv_filter_size = 4
        self.conv_groups = 4
        self.conv_padding = [1, 1]
        self.use_cudnn = True


if __name__ == "__main__":
    unittest.main()
