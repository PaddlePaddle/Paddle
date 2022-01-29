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
        self.bs = 2
        self.num_rois = 4
        self.channel = 8
        self.height = 16
        self.width = 16
        self.precision = AnalysisConfig.Precision.Float32
        self.serialize = False
        self.enable_trt = True

    def build(self):
        self.trt_parameters = TRTRoiAlignTest.TensorRTParam(
            1 << 30, self.bs * self.num_rois, 1, self.precision, self.serialize,
            False)
        with fluid.program_guard(self.main_program, self.startup_program):
            data_shape = [-1, self.channel, self.height, self.width]
            data = fluid.data(name='data', shape=data_shape, dtype='float32')
            rois = fluid.data(
                name='rois', shape=[-1, 4], dtype='float32', lod_level=1)
            roi_align_out = fluid.layers.roi_align(data, rois)
            out = fluid.layers.batch_norm(roi_align_out, is_test=True)

        rois_lod = fluid.create_lod_tensor(
            np.random.random([self.bs * self.num_rois, 4]).astype('float32'),
            [[self.num_rois, self.num_rois]], fluid.CPUPlace())

        data_shape[0] = self.bs
        self.feeds = {
            'data': np.random.random(data_shape).astype('float32'),
            'rois': rois_lod,
        }
        self.fetch_list = [out]

    def check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            atol = 1e-5
            if self.trt_parameters.precision == AnalysisConfig.Precision.Half:
                atol = 1e-3
            self.check_output_with_option(use_gpu, atol, flatten=True)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))

    def set_dynamic(self):
        min_shape_spec = dict()
        max_shape_spec = dict()
        opt_shape_spec = dict()
        min_shape_spec['data'] = [
            self.bs, self.channel, self.height // 2, self.width // 2
        ]
        min_shape_spec['rois'] = [1, 4]
        max_shape_spec[
            'data'] = [self.bs, self.channel, self.height * 2, self.width * 2]
        max_shape_spec['rois'] = [self.bs * self.num_rois, 4]
        opt_shape_spec[
            'data'] = [self.bs, self.channel, self.height, self.width]
        opt_shape_spec['rois'] = [self.bs * self.num_rois, 4]

        self.dynamic_shape_params = InferencePassTest.DynamicShapeParam(
            min_shape_spec, max_shape_spec, opt_shape_spec, False)

    def run_test(self):
        self.build()
        self.check_output()

    def test_base(self):
        self.run_test()

    def test_fp16(self):
        self.precision = AnalysisConfig.Precision.Half
        self.run_test()

    def test_serialize(self):
        self.serialize = True
        self.run_test()

    def test_dynamic(self):
        self.set_dynamic()
        self.run_test()

    def test_dynamic_fp16(self):
        self.set_dynamic()
        self.precision = AnalysisConfig.Precision.Half
        self.run_test()

    def test_dynamic_serialize(self):
        self.set_dynamic()
        self.serialize = True
        self.run_test()


if __name__ == "__main__":
    unittest.main()
