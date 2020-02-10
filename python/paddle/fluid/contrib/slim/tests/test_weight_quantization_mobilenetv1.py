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
import os
import time
from paddle.dataset.common import download, DATA_HOME
from paddle.fluid.contrib.slim.quantization import WeightQuantization


class TestWeightQuantization(unittest.TestCase):
    def setUp(self):
        self.weight_quantization_dir = 'weight_quantization'
        self.cache_folder = os.path.join(DATA_HOME,
                                         self.weight_quantization_dir)

    def download_model(self, model_name, data_url, data_md5):
        download(data_url, self.weight_quantization_dir, data_md5)
        file_name = data_url.split('/')[-1]
        file_path = os.path.join(self.cache_folder, file_name)
        print(model_name + ' is downloaded at ' + file_path)

        unziped_path = os.path.join(self.cache_folder, model_name)
        self.cache_unzipping(unziped_path, file_path)
        print(model_name + ' is unziped at ' + unziped_path)
        return unziped_path

    def cache_unzipping(self, target_folder, zip_path):
        if not os.path.exists(target_folder):
            cmd = 'mkdir {0} && tar xf {1} -C {0}'.format(target_folder,
                                                          zip_path)
            os.system(cmd)

    def run_test(self, model_name, model_data_url, model_data_md5, weight_bits,
                 quantizable_op_type, threshold_rate):

        model_dir = self.download_model(model_name, model_data_url,
                                        model_data_md5)

        timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        save_model_dir = os.path.join(
            os.getcwd(),
            model_name + "_wq_" + str(weight_bits) + "_" + timestamp)
        weight_quant = WeightQuantization(model_dir=model_dir + "/model")
        weight_quant.quantize_weight_to_int(
            save_model_dir=save_model_dir,
            weight_bits=weight_bits,
            quantizable_op_type=quantizable_op_type,
            threshold_rate=threshold_rate)
        print("finish weight quantization for " + model_name + "\n")

        try:
            os.system("rm -rf {}".format(save_model_dir))
        except Exception as e:
            print("Failed to delete {} due to {}".format(save_model_dir, str(
                e)))


class TestWeightQuantizationMobilenetv1(TestWeightQuantization):
    model_name = "mobilenetv1"
    model_data_url = "http://paddle-inference-dist.bj.bcebos.com/int8/mobilenetv1_int8_model.tar.gz"
    model_data_md5 = "13892b0716d26443a8cdea15b3c6438b"

    def test_weight_quantization_mobilenetv1_8bit(self):
        weight_bits = 8
        quantizable_op_type = ['conv2d', 'depthwise_conv2d', 'mul']
        threshold_rate = 0.0
        self.run_test(self.model_name, self.model_data_url, self.model_data_md5,
                      weight_bits, quantizable_op_type, threshold_rate)

    def test_weight_quantization_mobilenetv1_16bit(self):
        weight_bits = 16
        quantizable_op_type = ['conv2d', 'depthwise_conv2d', 'mul']
        threshold_rate = 1e-9
        self.run_test(self.model_name, self.model_data_url, self.model_data_md5,
                      weight_bits, quantizable_op_type, threshold_rate)


if __name__ == '__main__':
    unittest.main()
