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
    def download_model(self):
        # resnet50 fp32 data
        data_urls = [
            'http://paddle-inference-dist.bj.bcebos.com/int8/resnet50_int8_model.tar.gz'
        ]
        data_md5s = ['4a5194524823d9b76da6e738e1367881']
        self.model_cache_folder = self.download_data(data_urls, data_md5s,
                                                     "resnet50_fp32")
        self.model = "ResNet-50"
        self.algo = "KL"

    def test_post_training_resnet50(self):
        self.download_model()

        print("Start FP32 inference for {0} on {1} images ...".format(
            self.model, self.infer_iterations * self.batch_size))
        (fp32_throughput, fp32_latency,
         fp32_acc1) = self.run_program(self.model_cache_folder + "/model")

        print("Start INT8 post training quantization for {0} on {1} images ...".
              format(self.model, self.sample_iterations * self.batch_size))
        self.generate_quantized_model(
            self.model_cache_folder + "/model",
            algo=self.algo,
            is_full_quantize=False)

        print("Start INT8 inference for {0} on {1} images ...".format(
            self.model, self.infer_iterations * self.batch_size))
        (int8_throughput, int8_latency,
         int8_acc1) = self.run_program(self.int8_model)

        print(
            "FP32 {0}: batch_size {1}, throughput {2} images/second, latency {3} second, accuracy {4}".
            format(self.model, self.batch_size, fp32_throughput, fp32_latency,
                   fp32_acc1))
        print(
            "INT8 {0}: batch_size {1}, throughput {2} images/second, latency {3} second, accuracy {4}".
            format(self.model, self.batch_size, int8_throughput, int8_latency,
                   int8_acc1))
        sys.stdout.flush()

        delta_value = fp32_acc1 - int8_acc1
        self.assertLess(delta_value, 0.025)


if __name__ == '__main__':
    unittest.main()
