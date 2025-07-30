# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from openvino_test_base import OpenVINOBaseTest

import paddle
from paddle.static import InputSpec


class TestOpenVINOResnet50Model(OpenVINOBaseTest):
    def setUp(self):
        model = paddle.vision.models.resnet50(True)
        self.model_name = 'resnet50'
        input_spec = [InputSpec(shape=[None, 3, 224, 224], name='x')]
        self.to_static(model, input_spec)
        self.batch_size = 1
        self.infer_threads = 3
        self.precision = "float32"

    def test_model(self):
        self.check_result()


class TestOpenVINOResnet50ModelFP16(OpenVINOBaseTest):
    def setUp(self):
        model = paddle.vision.models.resnet50(True)
        self.model_name = 'resnet50'
        input_spec = [InputSpec(shape=[None, 3, 224, 224], name='x')]
        self.to_static(model, input_spec)
        self.batch_size = 2
        self.infer_threads = 3
        self.precision = "float16"

    def test_model(self):
        self.check_result()


class TestOpenVINOResnet50ModelBs2(OpenVINOBaseTest):
    def setUp(self):
        model = paddle.vision.models.resnet50(True)
        self.model_name = 'resnet50'
        input_spec = [InputSpec(shape=[None, 3, 224, 224], name='x')]
        self.to_static(model, input_spec)
        self.batch_size = 2
        self.infer_threads = 3
        self.precision = "float32"

    def test_model(self):
        self.check_result()


if __name__ == "__main__":
    unittest.main()
