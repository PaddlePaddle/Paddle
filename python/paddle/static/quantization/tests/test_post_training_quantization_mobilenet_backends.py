#   copyright (c) 2022 paddlepaddle authors. all rights reserved.
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

from test_post_training_quantization_mobilenetv1 import (
    TestPostTrainingQuantization,
)

import paddle

paddle.enable_static()


class TestPostTrainingAvgONNXFormatForMobilenetv1TensorRT(
    TestPostTrainingQuantization
):
    def test_post_training_onnx_format_mobilenetv1_tensorrt(self):
        model = "MobileNet-V1"
        algo = "avg"
        round_type = "round"
        data_urls = [
            'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV1_infer.tar'
        ]
        data_md5s = ['5ee2b1775b11dc233079236cdc216c2e']
        quantizable_op_type = [
            "conv2d",
            "depthwise_conv2d",
            "mul",
        ]
        is_full_quantize = False
        is_use_cache_file = False
        is_optimize_model = False
        onnx_format = True
        diff_threshold = 0.05
        batch_nums = 10
        deploy_backend = "tensorrt"
        self.run_test(
            model,
            'inference.pdmodel',
            'inference.pdiparams',
            algo,
            round_type,
            data_urls,
            data_md5s,
            quantizable_op_type,
            is_full_quantize,
            is_use_cache_file,
            is_optimize_model,
            diff_threshold,
            onnx_format=onnx_format,
            batch_nums=batch_nums,
            deploy_backend=deploy_backend,
        )


class TestPostTrainingKLONNXFormatForMobilenetv1MKLDNN(
    TestPostTrainingQuantization
):
    def test_post_training_onnx_format_mobilenetv1_mkldnn(self):
        model = "MobileNet-V1"
        algo = "KL"
        round_type = "round"
        data_urls = [
            'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV1_infer.tar'
        ]
        data_md5s = ['5ee2b1775b11dc233079236cdc216c2e']
        quantizable_op_type = [
            "conv2d",
            "depthwise_conv2d",
            "mul",
        ]
        is_full_quantize = False
        is_use_cache_file = False
        is_optimize_model = False
        onnx_format = True
        diff_threshold = 0.05
        batch_nums = 2
        deploy_backend = "mkldnn"
        self.run_test(
            model,
            'inference.pdmodel',
            'inference.pdiparams',
            algo,
            round_type,
            data_urls,
            data_md5s,
            quantizable_op_type,
            is_full_quantize,
            is_use_cache_file,
            is_optimize_model,
            diff_threshold,
            onnx_format=onnx_format,
            batch_nums=batch_nums,
            deploy_backend=deploy_backend,
        )


class TestPostTrainingAvgONNXFormatForMobilenetv1ARMCPU(
    TestPostTrainingQuantization
):
    def test_post_training_onnx_format_mobilenetv1_armcpu(self):
        model = "MobileNet-V1"
        algo = "avg"
        round_type = "round"
        data_urls = [
            'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV1_infer.tar'
        ]
        data_md5s = ['5ee2b1775b11dc233079236cdc216c2e']
        quantizable_op_type = [
            "conv2d",
            "depthwise_conv2d",
            "mul",
        ]
        is_full_quantize = False
        is_use_cache_file = False
        is_optimize_model = True
        onnx_format = True
        diff_threshold = 0.05
        batch_nums = 3
        deploy_backend = "arm"
        self.run_test(
            model,
            'inference.pdmodel',
            'inference.pdiparams',
            algo,
            round_type,
            data_urls,
            data_md5s,
            quantizable_op_type,
            is_full_quantize,
            is_use_cache_file,
            is_optimize_model,
            diff_threshold,
            onnx_format=onnx_format,
            batch_nums=batch_nums,
            deploy_backend=deploy_backend,
        )


if __name__ == '__main__':
    unittest.main()
