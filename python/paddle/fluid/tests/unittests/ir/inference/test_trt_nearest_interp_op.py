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
from inference_pass_test import InferencePassTest

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.static.nn as nn
from paddle.fluid.core import AnalysisConfig, PassVersionChecker


class TRTNearestInterpTest(InferencePassTest):
    def setUp(self):
        self.set_params()

        with fluid.program_guard(self.main_program, self.startup_program):
            if self.data_layout == 'NCHW':
                shape = [
                    -1,
                    self.channels,
                    self.origin_shape[0],
                    self.origin_shape[1],
                ]
            else:
                shape = [
                    -1,
                    self.origin_shape[0],
                    self.origin_shape[1],
                    self.channels,
                ]
            data = fluid.data(name='data', shape=shape, dtype='float32')
            resize_out = self.append_nearest_interp(data)
            out = nn.batch_norm(resize_out, is_test=True)

        if self.data_layout == 'NCHW':
            shape = [
                self.bs,
                self.channels,
                self.origin_shape[0],
                self.origin_shape[1],
            ]
        else:
            shape = [
                self.bs,
                self.origin_shape[0],
                self.origin_shape[1],
                self.channels,
            ]

        self.feeds = {
            'data': np.random.random(shape).astype('float32'),
        }
        self.enable_trt = True
        self.trt_parameters = TRTNearestInterpTest.TensorRTParam(
            1 << 30, self.bs, 1, AnalysisConfig.Precision.Float32, False, False
        )
        self.fetch_list = [out]

    def set_params(self):
        self.bs = 4
        self.scale = 0
        self.channels = 3

        self.origin_shape = (4, 4)  # HW
        self.resize_shape = (16, 16)  # HW
        self.align_corners = True
        self.data_layout = 'NCHW'

    def append_nearest_interp(self, data):
        if self.scale > 0.0:
            return paddle.nn.functional.interpolate(
                data,
                scale_factor=self.scale,
                data_format=self.data_layout,
            )
        return paddle.nn.functional.interpolate(
            data,
            size=self.resize_shape,
            data_format=self.data_layout,
        )

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, flatten=True)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass')
            )


class TRTNearestInterpTest1(TRTNearestInterpTest):
    def set_params(self):
        self.bs = 4
        self.scale = -1
        self.channels = 3
        self.origin_shape = (16, 16)  # HW
        self.resize_shape = (32, 32)  # HW
        self.align_corners = True
        self.data_layout = 'NCHW'


class TRTNearestInterpTest2(TRTNearestInterpTest):
    def set_params(self):
        self.bs = 4
        self.scale = 2.0
        self.channels = 3
        self.origin_shape = (16, 16)  # HW
        self.resize_shape = (32, 32)  # HW
        self.align_corners = False
        self.data_layout = 'NCHW'


class TRTNearestInterpTest3(TRTNearestInterpTest):
    def set_params(self):
        self.bs = 4
        self.scale = 0
        self.channels = 3
        self.origin_shape = (16, 16)  # HW
        self.resize_shape = (32, 32)  # HW
        self.align_corners = False
        self.data_layout = 'NCHW'


class TRTNearestInterpTest4(TRTNearestInterpTest):
    def set_params(self):
        self.bs = 4
        self.scale = -1
        self.channels = 3
        self.origin_shape = (16, 16)  # HW
        self.resize_shape = (47, 12)  # HW
        self.align_corners = False
        self.data_layout = 'NCHW'


class TRTNearestInterpTest5(TRTNearestInterpTest):
    def set_params(self):
        self.bs = 4
        self.scale = -1
        self.channels = 3
        self.origin_shape = (16, 16)  # HW
        self.resize_shape = (32, 32)  # HW
        self.align_corners = True
        self.data_layout = 'NHWC'


class TRTNearestInterpTest6(TRTNearestInterpTest):
    def set_params(self):
        self.bs = 4
        self.scale = 2.0
        self.channels = 3
        self.origin_shape = (16, 16)  # HW
        self.resize_shape = (32, 32)  # HW
        self.align_corners = False
        self.data_layout = 'NHWC'


class TRTNearestInterpTest7(TRTNearestInterpTest):
    def set_params(self):
        self.bs = 4
        self.scale = -1
        self.channels = 3
        self.origin_shape = (16, 16)  # HW
        self.resize_shape = (32, 32)  # HW
        self.align_corners = False
        self.data_layout = 'NHWC'


class TRTNearestInterpTest8(TRTNearestInterpTest):
    def set_params(self):
        self.bs = 4
        self.scale = -1
        self.channels = 3
        self.origin_shape = (16, 16)  # HW
        self.resize_shape = (47, 12)  # HW
        self.align_corners = False
        self.data_layout = 'NHWC'


class TRTNearestInterpTest9(TRTNearestInterpTest):
    def set_params(self):
        self.bs = 4
        self.scale = -1
        self.channels = 3
        self.origin_shape = (16, 16)  # HW
        self.resize_shape = (47, 12)  # HW
        self.align_corners = False
        self.data_layout = 'NHWC'


if __name__ == "__main__":
    unittest.main()
