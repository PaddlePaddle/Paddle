#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np

from inference_test_ipu import IPUInferenceTest


class TestErnie(IPUInferenceTest):

    def setUp(self):
        self.set_threshold(mae_fp32=3e-7,
                           mse_fp32=2e-13,
                           mae_fp16=4e-4,
                           mse_fp16=2e-7)
        self.set_model("ernie")
        self.set_batch_size(1)
        self.set_fp16(True)
        self.set_data_feed()
        self.output_dict = {}

    def set_data_feed(self):
        data1 = np.random.randint(low=0,
                                  high=18000,
                                  size=(self.batch_size, 128,
                                        1)).astype(np.int64)
        data2 = np.random.randint(low=0,
                                  high=513,
                                  size=(self.batch_size, 128,
                                        1)).astype(np.int64)
        data3 = np.random.randint(low=0, high=3, size=(self.batch_size, 128,
                                                       1)).astype(np.int64)
        data4 = np.random.uniform(size=[self.batch_size, 128, 1])
        self.feed_fp32 = [data1, data2, data3, data4.astype(np.float32)]
        self.feed_fp16 = [data1, data2, data3, data4.astype(np.float16)]
        self.data_shape = [data1.shape, data2.shape, data3.shape, data4.shape]

    def test(self):
        for m in IPUInferenceTest.ExecutionMode:
            if not self.fp16_mode and m == IPUInferenceTest.ExecutionMode.IPU_FP16:
                continue
            self.create_predictor(m)
            self.run_model(m)
        self.check()


class TestErnieMultiBatch(TestErnie):

    def setUp(self):
        self.set_threshold(mae_fp32=2e-7,
                           mse_fp32=5e-14,
                           mae_fp16=6e-4,
                           mse_fp16=4e-7)
        self.set_model("ernie")
        self.set_batch_size(2)
        self.set_fp16(True)
        self.set_data_feed()
        self.output_dict = {}


if __name__ == '__main__':
    unittest.main()
