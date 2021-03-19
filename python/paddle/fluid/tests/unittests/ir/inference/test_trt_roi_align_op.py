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


class TRTRoiAlignTest(InferencePassTest):
    def setUp(self):
        self.set_params()
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name='data', shape=[-1, 32, 64, 64], dtype='float32')
            rois = fluid.data(
                name='rois', shape=[-1, 4], dtype='float32', lod_level=1)
            roi_align_out = self.append_roi_align(data, rois)
            roi_align_out = fluid.layers.reshape(roi_align_out,
                                                 [self.num_rois, 32])
            out = fluid.layers.batch_norm(roi_align_out, is_test=True)

        rois_lod = fluid.create_lod_tensor(
            np.random.random([self.num_rois, 4]).astype('float32'),
            [[0, self.num_rois]], fluid.CPUPlace())
        self.feeds = {
            'data': np.random.random([self.bs, 32, 64, 64]).astype('float32'),
            'rois': rois_lod,
        }
        self.enable_trt = True
        self.trt_parameters = TRTRoiAlignTest.TensorRTParam(
            1 << 30, self.bs, 1, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def set_params(self):
        self.bs = 2
        self.num_rois = 2

    def append_roi_align(self, data, rois):
        return fluid.layers.roi_align(data, rois)

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, flatten=True)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))


if __name__ == "__main__":
    unittest.main()
