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

from __future__ import print_function

import unittest
import numpy as np
from inference_pass_test import InferencePassTest
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.core import PassVersionChecker
from paddle.fluid.core import AnalysisConfig


class TRTBilinearInterpTest(InferencePassTest):
    def setUp(self):
        self.set_params()

        with fluid.program_guard(self.main_program, self.startup_program):
            if self.data_layout == 'NCHW':
                shape = [
                    -1, self.channels, self.origin_shape[0],
                    self.origin_shape[1]
                ]
            else:
                shape = [
                    -1, self.origin_shape[0], self.origin_shape[1],
                    self.channels
                ]
            data = fluid.data(name='data', shape=shape, dtype='float32')
            resize_out = self.append_bilinear_interp(data)
            out = fluid.layers.batch_norm(resize_out, is_test=True)

        if self.data_layout == 'NCHW':
            shape = [
                self.bs, self.channels, self.origin_shape[0],
                self.origin_shape[1]
            ]
        else:
            shape = [
                self.bs, self.origin_shape[0], self.origin_shape[1],
                self.channels
            ]

        self.feeds = {'data': np.random.random(shape).astype('float32'), }
        self.enable_trt = True
        self.trt_parameters = TRTBilinearInterpTest.TensorRTParam(
            1 << 30, self.bs, 1, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def set_params(self):
        self.bs = 4
        self.scale = 1
        self.channels = 3
        self.origin_shape = (32, 32)  # HW
        self.resize_shape = (64, 64)  # HW
        self.align_corners = True
        self.data_layout = 'NCHW'

    def append_bilinear_interp(self, data):
        if self.scale > 0.:
            return fluid.layers.resize_bilinear(
                data,
                scale=self.scale,
                align_corners=self.align_corners,
                data_format=self.data_layout)
        return fluid.layers.resize_bilinear(
            data,
            out_shape=self.resize_shape,
            align_corners=self.align_corners,
            data_format=self.data_layout)

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.enable_mkldnn = True
            self.check_output_with_option(use_gpu, flatten=True)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))


class TRTBilinearInterpTest1(TRTBilinearInterpTest):
    def set_params(self):
        self.bs = 1
        self.scale = -1
        self.channels = 2
        self.origin_shape = (6, 4)  # HW
        self.resize_shape = (13, 12)  # HW
        self.align_corners = True
        self.align_mode = 0
        self.data_layout = 'NCHW'


class TRTBilinearInterpTest2(TRTBilinearInterpTest):
    def set_params(self):
        self.bs = 4
        self.scale = 2.
        self.channels = 3
        self.origin_shape = (32, 32)  # HW
        self.resize_shape = (64, 64)  # HW
        self.align_corners = False
        self.data_layout = 'NCHW'


class TRTBilinearInterpTest3(TRTBilinearInterpTest):
    def set_params(self):
        self.bs = 4
        self.scale = -1
        self.channels = 3
        self.origin_shape = (32, 32)  # HW
        self.resize_shape = (64, 64)  # HW
        self.align_corners = False
        self.data_layout = 'NCHW'


class TRTBilinearInterpTest4(TRTBilinearInterpTest):
    def set_params(self):
        self.bs = 4
        self.scale = -1
        self.channels = 3
        self.origin_shape = (32, 32)  # HW
        self.resize_shape = (47, 48)  # HW
        self.align_corners = False
        self.data_layout = 'NCHW'


class TRTBilinearInterpTest5(TRTBilinearInterpTest):
    def set_params(self):
        self.bs = 4
        self.scale = -1
        self.channels = 3
        self.origin_shape = (32, 32)  # HW
        self.resize_shape = (64, 64)  # HW
        self.align_corners = False
        self.data_layout = 'NHWC'


class TRTBilinearInterpTest6(TRTBilinearInterpTest):
    def set_params(self):
        self.bs = 4
        self.scale = 2.
        self.channels = 3
        self.origin_shape = (32, 32)  # HW
        self.resize_shape = (64, 64)  # HW
        self.align_corners = False
        self.data_layout = 'NHWC'


class TRTBilinearInterpTest7(TRTBilinearInterpTest):
    def set_params(self):
        self.bs = 4
        self.scale = -1
        self.channels = 3
        self.origin_shape = (32, 32)  # HW
        self.resize_shape = (64, 64)  # HW
        self.align_corners = False
        self.data_layout = 'NHWC'


class TRTBilinearInterpTest8(TRTBilinearInterpTest):
    def set_params(self):
        self.bs = 4
        self.scale = -1
        self.channels = 3
        self.origin_shape = (32, 32)  # HW
        self.resize_shape = (47, 48)  # HW
        self.align_corners = False
        self.data_layout = 'NHWC'


class TRTBilinearInterpTest9(TRTBilinearInterpTest):
    def set_params(self):
        self.bs = 4
        self.scale = -1
        self.channels = 3
        self.origin_shape = (32, 32)  # HW
        self.resize_shape = (47, 48)  # HW
        self.align_corners = False
        self.data_layout = 'NHWC'


class TRTBilinearInterpTest10(TRTBilinearInterpTest):
    def set_params(self):
        self.bs = 4
        self.scale = -1
        self.channels = 3
        self.origin_shape = (32, 32)  # HW
        self.resize_shape = (64, 64)  # HW
        self.align_corners = False
        self.align_mode = 0
        self.data_layout = 'NHWC'


class TRTBilinearInterpTest11(TRTBilinearInterpTest):
    def set_params(self):
        self.bs = 4
        self.scale = -3
        self.channels = 3
        self.origin_shape = (32, 32)  # HW
        self.resize_shape = (47, 48)  # HW
        self.align_corners = False
        self.align_mode = 0
        self.data_layout = 'NHWC'


class TRTBilinearInterpTest12(TRTBilinearInterpTest):
    def set_params(self):
        self.bs = 4
        self.scale = 2
        self.channels = 3
        self.origin_shape = (32, 32)  # HW
        self.resize_shape = (47, 48)  # HW
        self.align_corners = False
        self.align_mode = 0
        self.data_layout = 'NHWC'


if __name__ == "__main__":
    unittest.main()
