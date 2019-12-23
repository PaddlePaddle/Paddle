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


class TestPostTrainingForResnet50(TestPostTrainingQuantization):
    def test_post_training_resnet50(self):
        model = "ResNet-50"
        algo = "direct"
        data_urls = [
            'http://paddle-inference-dist.bj.bcebos.com/int8/resnet50_int8_model.tar.gz'
        ]
        data_md5s = ['4a5194524823d9b76da6e738e1367881']
        is_full_quantize = False
        is_use_cache_file = True
        self.run_test(model, algo, data_urls, data_md5s, is_full_quantize,
                      is_use_cache_file)


if __name__ == '__main__':
    unittest.main()
