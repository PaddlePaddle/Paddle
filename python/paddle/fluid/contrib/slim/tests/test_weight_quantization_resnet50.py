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

import unittest
from test_weight_quantization_mobilenetv1 import TestWeightQuantization


class TestWeightQuantizationResnet50(TestWeightQuantization):
    model_name = "resnet50"
    model_data_url = "http://paddle-inference-dist.bj.bcebos.com/int8/resnet50_int8_model.tar.gz"
    model_data_md5 = "4a5194524823d9b76da6e738e1367881"

    def test_weight_quantization_resnet50_8bit(self):
        quantize_weight_bits = 8
        quantizable_op_type = ['conv2d', 'depthwise_conv2d', 'mul']
        threshold_rate = 0.0
        self.run_test(self.model_name, self.model_data_url, self.model_data_md5,
                      quantize_weight_bits, quantizable_op_type, threshold_rate)

    def test_weight_quantization_resnet50_16bit(self):
        quantize_weight_bits = 16
        quantizable_op_type = ['conv2d', 'depthwise_conv2d', 'mul']
        threshold_rate = 0.0
        self.run_test(self.model_name, self.model_data_url, self.model_data_md5,
                      quantize_weight_bits, quantizable_op_type, threshold_rate)


if __name__ == '__main__':
    unittest.main()
