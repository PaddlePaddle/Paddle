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
import paddle.nn.functional as fun
from paddle.fluid.core import PassVersionChecker
from paddle.fluid.core import AnalysisConfig


class TRTBilinearInterpV2Test(InferencePassTest):
    def setUp(self):
        self.set_params()

        with fluid.program_guard(self.main_program, self.startup_program):
            if self.data_layout == 'NCHW':
                shape = [
                    1, self.channels, self.origin_shape[0],
                    self.origin_shape[1]
                ]
            else:
                shape = [
                    1, self.origin_shape[0], self.origin_shape[1],
                    self.channels
                ]
            data = fluid.data(name='data', shape=shape, dtype='float32')
            resize_out = self.append_bilinear_interp_v2(data)
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
        self.trt_parameters = TRTBilinearInterpV2Test.TensorRTParam(
            1 << 30, self.bs, 1, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def set_params(self):
        self.bs = 1 
        self.channels = 3
        self.origin_shape = (32, 32)  # HW
        self.resize_shape = (64, 64)  # HW
        self.align_corners = False
        self.data_layout = 'NCHW'

    def append_bilinear_interp_v2(self, data):
        x = np.array([0.5, 0.5, 0.5]).astype("float32")
        scale_tensor=fluid.data(name="scale", shape=[3], dtype="float32")
        fluid.layers.assign(x, scale_tensor)
        return fun.interpolate(x=data, scale_factor=[2,1], mode="bilinear")

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.enable_mkldnn = True
            self.check_output_with_option(use_gpu, flatten=True)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))


class TRTBilinearInterpV2Test1(TRTBilinearInterpV2Test):
    def set_params(self):
        self.bs = 1
        self.channels = 3 
        self.origin_shape = (6, 4)  # HW
        self.resize_shape = (18, 12)  # HW
        self.align_corners = False 
        self.align_mode = 0
        self.data_layout = 'NCHW'


if __name__ == "__main__":
    unittest.main()
