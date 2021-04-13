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


class TRTYoloBoxTest(InferencePassTest):
    def setUp(self):
        self.set_params()
        with fluid.program_guard(self.main_program, self.startup_program):
            image_shape = [self.bs, self.channel, self.height, self.width]
            image = fluid.data(name='image', shape=image_shape, dtype='float32')
            image_size = fluid.data(
                name='image_size', shape=[self.bs, 2], dtype='int32')
            boxes, scores = self.append_yolobox(image, image_size)
            scores = fluid.layers.reshape(scores, (self.bs, -1))
            out = fluid.layers.batch_norm(scores, is_test=True)

        self.feeds = {
            'image': np.random.random(image_shape).astype('float32'),
            'image_size': np.random.randint(
                32, 64, size=(self.bs, 2)).astype('int32'),
        }
        self.enable_trt = True
        self.trt_parameters = TRTYoloBoxTest.TensorRTParam(
            1 << 30, self.bs, 1, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out, boxes]

    def set_params(self):
        self.bs = 4
        self.channel = 255
        self.height = 64
        self.width = 64
        self.class_num = 80
        self.anchors = [10, 13, 16, 30, 33, 23]
        self.conf_thresh = .1
        self.downsample_ratio = 32

    def append_yolobox(self, image, image_size):
        return fluid.layers.yolo_box(
            x=image,
            img_size=image_size,
            class_num=self.class_num,
            anchors=self.anchors,
            conf_thresh=self.conf_thresh,
            downsample_ratio=self.downsample_ratio)

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, flatten=True)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))


if __name__ == "__main__":
    unittest.main()
