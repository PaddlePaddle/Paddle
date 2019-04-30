#   copyright (c) 2018 paddlepaddle authors. all rights reserved.
#
# licensed under the apache license, version 2.0 (the "license");
# you may not use this file except in compliance with the license.
# you may obtain a copy of the license at
#
#     http://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "as is" basis,
# without warranties or conditions of any kind, either express or implied.
# see the license for the specific language governing permissions and
# limitations under the license.
import unittest
import sys
from test_calibration_resnet50 import TestCalibration


class TestCalibrationForMobilenetv1(TestCalibration):
    def download_model(self):
        # mobilenetv1 fp32 data
        data_urls = [
            'http://paddle-inference-dist.bj.bcebos.com/int8/mobilenetv1_int8_model.tar.gz'
        ]
        data_md5s = ['13892b0716d26443a8cdea15b3c6438b']
        self.model_cache_folder = self.download_data(data_urls, data_md5s,
                                                     "mobilenetv1_fp32")
        self.model = "MobileNet-V1"
        self.algo = "KL"

    def test_calibration(self):
        self.download_model()
        print("Start FP32 inference for {0} on {1} images ...").format(
            self.model, self.infer_iterations * self.batch_size)
        (fp32_throughput, fp32_latency,
         fp32_acc1) = self.run_program(self.model_cache_folder + "/model")
        print("Start INT8 calibration for {0} on {1} images ...").format(
            self.model, self.sample_iterations * self.batch_size)
        self.run_program(
            self.model_cache_folder + "/model", True, algo=self.algo)
        print("Start INT8 inference for {0} on {1} images ...").format(
            self.model, self.infer_iterations * self.batch_size)
        (int8_throughput, int8_latency,
         int8_acc1) = self.run_program(self.int8_model)
        delta_value = fp32_acc1 - int8_acc1
        self.assertLess(delta_value, 0.01)
        print(
            "FP32 {0}: batch_size {1}, throughput {2} images/second, latency {3} second, accuracy {4}".
            format(self.model, self.batch_size, fp32_throughput, fp32_latency,
                   fp32_acc1))
        print(
            "INT8 {0}: batch_size {1}, throughput {2} images/second, latency {3} second, accuracy {4}".
            format(self.model, self.batch_size, int8_throughput, int8_latency,
                   int8_acc1))
        sys.stdout.flush()


if __name__ == '__main__':
    unittest.main()
