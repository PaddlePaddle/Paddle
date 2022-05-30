# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import unittest
from test_post_training_quantization_mobilenetv1 import TestPostTrainingQuantization
import paddle

paddle.enable_static()


class TestPostTrainingForResnet50(TestPostTrainingQuantization):
    def test_post_training_resnet50(self):
        model = "ResNet-50"
        algo = "min_max"
        round_type = "round"
        data_urls = [
            'http://paddle-inference-dist.bj.bcebos.com/int8/resnet50_int8_model.tar.gz'
        ]
        data_md5s = ['4a5194524823d9b76da6e738e1367881']
        quantizable_op_type = ["conv2d", "mul"]
        is_full_quantize = False
        is_use_cache_file = False
        is_optimize_model = False
        diff_threshold = 0.025
        self.run_test(model, algo, round_type, data_urls, data_md5s,
                      quantizable_op_type, is_full_quantize, is_use_cache_file,
                      is_optimize_model, diff_threshold)


class TestPostTrainingForResnet50ONNXFormat(TestPostTrainingQuantization):
    def test_post_training_resnet50(self):
        model = "ResNet-50"
        algo = "min_max"
        round_type = "round"
        data_urls = [
            'http://paddle-inference-dist.bj.bcebos.com/int8/resnet50_int8_model.tar.gz'
        ]
        data_md5s = ['4a5194524823d9b76da6e738e1367881']
        quantizable_op_type = ["conv2d", "mul"]
        is_full_quantize = False
        is_use_cache_file = False
        is_optimize_model = False
        diff_threshold = 0.025
        onnx_format = True
        self.run_test(
            model,
            algo,
            round_type,
            data_urls,
            data_md5s,
            quantizable_op_type,
            is_full_quantize,
            is_use_cache_file,
            is_optimize_model,
            diff_threshold,
            onnx_format=onnx_format)


if __name__ == '__main__':
    unittest.main()
