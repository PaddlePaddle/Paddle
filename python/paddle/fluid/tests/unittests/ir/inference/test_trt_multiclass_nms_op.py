# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import itertools
import numpy as np
from inference_pass_test import InferencePassTest
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.core import PassVersionChecker
from paddle.fluid.core import AnalysisConfig


class TensorRTMultiClassNMSTest(InferencePassTest):
    def setUp(self):
        self.enable_trt = True
        self.enable_tensorrt_oss = True
        self.precision = AnalysisConfig.Precision.Float32
        self.serialize = False
        self.bs = 1
        self.background_label = -1
        self.score_threshold = .5
        self.nms_top_k = 8
        self.nms_threshold = .3
        self.keep_top_k = 8
        self.normalized = False
        self.num_classes = 8
        self.num_boxes = 8
        self.trt_parameters = InferencePassTest.TensorRTParam(
            1 << 30, self.bs, 2, self.precision, self.serialize, False)

    def build(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            boxes = fluid.data(
                name='bboxes', shape=[-1, self.num_boxes, 4], dtype='float32')
            scores = fluid.data(
                name='scores',
                shape=[-1, self.num_classes, self.num_boxes],
                dtype='float32')
            multiclass_nms_out = fluid.layers.multiclass_nms(
                bboxes=boxes,
                scores=scores,
                background_label=self.background_label,
                score_threshold=self.score_threshold,
                nms_top_k=self.nms_top_k,
                nms_threshold=self.nms_threshold,
                keep_top_k=self.keep_top_k,
                normalized=self.normalized)
            mutliclass_nms_out = multiclass_nms_out + 1.
            multiclass_nms_out = fluid.layers.reshape(
                multiclass_nms_out, [self.bs, 1, self.keep_top_k, 6],
                name='reshape')
            out = fluid.layers.batch_norm(multiclass_nms_out, is_test=True)

        boxes_data = np.arange(self.num_boxes * 4).reshape(
            [self.bs, self.num_boxes, 4]).astype('float32')
        scores_data = np.arange(1 * self.num_classes * self.num_boxes).reshape(
            [self.bs, self.num_classes, self.num_boxes]).astype('float32')
        self.feeds = {
            'bboxes': boxes_data,
            'scores': scores_data,
        }
        self.fetch_list = [out]

    def run_test(self):
        self.build()
        self.check_output()

    def run_test_all(self):
        precision_opt = [
            AnalysisConfig.Precision.Float32, AnalysisConfig.Precision.Half
        ]
        serialize_opt = [False, True]
        max_shape = {
            'bboxes': [self.bs, self.num_boxes, 4],
            'scores': [self.bs, self.num_classes, self.num_boxes],
        }
        opt_shape = max_shape
        dynamic_shape_opt = [
            None, InferencePassTest.DynamicShapeParam({
                'bboxes': [1, 1, 4],
                'scores': [1, 1, 1]
            }, max_shape, opt_shape, False)
        ]
        for precision, serialize, dynamic_shape in itertools.product(
                precision_opt, serialize_opt, dynamic_shape_opt):
            self.precision = precision
            self.serialize = serialize
            self.dynamic_shape_params = dynamic_shape
            self.build()
            self.check_output()

    def check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))

    def test_base(self):
        self.run_test()

    def test_fp16(self):
        self.precision = AnalysisConfig.Precision.Half
        self.run_test()

    def test_serialize(self):
        self.serialize = True
        self.run_test()

    def test_dynamic(self):
        max_shape = {
            'bboxes': [self.bs, self.num_boxes, 4],
            'scores': [self.bs, self.num_classes, self.num_boxes],
        }
        opt_shape = max_shape
        self.dynamic_shape_params = InferencePassTest.DynamicShapeParam({
            'bboxes': [1, 1, 4],
            'scores': [1, 1, 1]
        }, max_shape, opt_shape, False)
        self.run_test()

    def test_background(self):
        self.background = 7
        self.run_test()

    def test_disable_oss(self):
        self.diable_tensorrt_oss = False
        self.run_test()


if __name__ == "__main__":
    unittest.main()
