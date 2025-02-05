# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from inference_pass_test import InferencePassTest

import paddle
from paddle import base
from paddle.base import core
from paddle.base.core import AnalysisConfig, PassVersionChecker

os.environ['NVIDIA_TF32_OVERRIDE'] = '0'


class TRTDeformableConvTest(InferencePassTest):
    def setUp(self):
        self.set_params()
        with base.program_guard(self.main_program, self.startup_program):
            input = paddle.static.data(
                name='input', shape=self.input_size, dtype=self.dtype
            )
            offset = paddle.static.data(
                name='offset', shape=self.offset_size, dtype=self.dtype
            )
            mask = paddle.static.data(
                name='mask', shape=self.mask_size, dtype=self.dtype
            )

            output = paddle.static.nn.common.deformable_conv(
                input,
                offset,
                mask,
                self.num_filters,
                self.filter_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilations,
                groups=self.groups,
                deformable_groups=self.deformable_groups,
                im2col_step=self.im2col_step,
            )

        self.feeds = {
            'input': np.random.random(self.input_size).astype(self.dtype),
            'offset': np.random.random(self.offset_size).astype(self.dtype),
            'mask': np.random.random(self.mask_size).astype(self.dtype),
        }
        self.enable_trt = True
        dtype = AnalysisConfig.Precision.Float32
        if self.dtype == 'float16':
            dtype = AnalysisConfig.Precision.Half
        self.trt_parameters = TRTDeformableConvTest.TensorRTParam(
            1 << 30, self.bs, 0, dtype, False, False
        )
        self.fetch_list = [output]

    def set_params(self):
        self.groups = 1
        self.padding = [1, 1]
        self.dilations = [1, 1]
        self.stride = [1, 1]
        self.im2col_step = 1
        self.deformable_groups = 1

        self.bs = 2
        self.input_size = [self.bs, 8, 4, 4]
        self.num_filters = 8
        self.filter_size = 3
        offset_c = (
            2 * self.deformable_groups * self.filter_size * self.filter_size
        )
        mask_c = self.deformable_groups * self.filter_size * self.filter_size
        self.offset_size = [
            self.input_size[0],
            offset_c,
            self.input_size[2],
            self.input_size[3],
        ]
        self.mask_size = [
            self.input_size[0],
            mask_c,
            self.input_size[2],
            self.input_size[3],
        ]

        self.dtype = 'float32'

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass')
            )


if __name__ == "__main__":
    unittest.main()
