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
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.core import PassVersionChecker
from paddle.fluid.core import AnalysisConfig


class TensorRTMultiClassNMSTest(InferencePassTest):
    def setUp(self):
        self.set_params()
        with fluid.program_guard(self.main_program, self.startup_program):
            boxes = fluid.data(
                name='bboxes', shape=[1, self.num_boxes, 4], dtype='float32')
            scores = fluid.data(
                name='scores',
                shape=[1, self.num_classes, self.num_boxes],
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
            multiclass_nms_out = fluid.layers.reshape(
                multiclass_nms_out, [1, 1, self.keep_top_k, 6])
            out = fluid.layers.batch_norm(multiclass_nms_out, is_test=True)

        boxes_data = np.arange(self.num_boxes * 4).reshape(
            [1, self.num_boxes, 4]).astype("float32")
        scores_data = np.arange(1 * self.num_classes * self.num_boxes).reshape(
            [1, self.num_classes, self.num_boxes]).astype("float32")
        self.feeds = {
            "bboxes": boxes_data,
            "scores": scores_data,
        }
        self.enable_trt = True
        self.trt_parameters = TensorRTMultiClassNMSTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def set_params(self):
        self.enable_tensorrt_oss = True
        self.background_label = -1
        self.score_threshold = .5
        self.nms_top_k = 8
        self.nms_threshold = .3
        self.keep_top_k = 8
        self.normalized = False
        self.num_classes = 8
        self.num_boxes = 8

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))


class TensorRTMultiClassNMSTest1(TensorRTMultiClassNMSTest):
    def set_params(self):
        self.enable_tensorrt_oss = True
        self.background_label = -1
        self.score_threshold = .5
        self.nms_top_k = 16
        self.nms_threshold = .3
        self.keep_top_k = 16
        self.normalized = False
        self.num_classes = 8
        self.num_boxes = 16


class TensorRTMultiClassNMSTest2(TensorRTMultiClassNMSTest):
    def set_params(self):
        self.enable_tensorrt_oss = True
        self.background_label = 7
        self.score_threshold = .5
        self.nms_top_k = 8
        self.nms_threshold = .3
        self.keep_top_k = 8
        self.normalized = False
        self.num_classes = 8
        self.num_boxes = 8


class TensorRTMultiClassNMSTest3(TensorRTMultiClassNMSTest):
    def set_params(self):
        self.enable_tensorrt_oss = False
        self.background_label = -1
        self.score_threshold = .5
        self.nms_top_k = 16
        self.nms_threshold = .3
        self.keep_top_k = 16
        self.normalized = False
        self.num_classes = 8
        self.num_boxes = 16


if __name__ == "__main__":
    unittest.main()
