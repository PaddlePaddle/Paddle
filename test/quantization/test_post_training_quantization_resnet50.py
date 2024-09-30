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
import time
import unittest

import numpy as np
from test_post_training_quantization_mobilenetv1 import (
    TestPostTrainingQuantization,
    val,
)

import paddle

paddle.enable_static()


class TestPostTrainingForResnet50(TestPostTrainingQuantization):
    def test_post_training_resnet50(self):
        model = "ResNet-50"
        algo = "min_max"
        round_type = "round"
        data_urls = [
            'http://paddle-inference-dist.bj.bcebos.com/int8/resnet50_int8_model_combined.tar.gz'
        ]
        data_md5s = ['db212fd4e9edc83381aef4533107e60c']
        quantizable_op_type = ["conv2d", "mul"]
        is_full_quantize = False
        is_use_cache_file = False
        is_optimize_model = False
        diff_threshold = 0.025
        self.run_test(
            model,
            'model.pdmodel',
            'model.pdiparams',
            algo,
            round_type,
            data_urls,
            data_md5s,
            "model",
            quantizable_op_type,
            is_full_quantize,
            is_use_cache_file,
            is_optimize_model,
            diff_threshold,
        )

    def run_program(
        self,
        model_path,
        model_filename,
        params_filename,
        batch_size,
        infer_iterations,
    ):
        image_shape = [3, 224, 224]
        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        [
            infer_program,
            feed_dict,
            fetch_targets,
        ] = paddle.static.load_inference_model(
            model_path,
            exe,
            model_filename=model_filename,
            params_filename=params_filename,
        )
        val_reader = paddle.batch(val(), batch_size)
        iterations = infer_iterations

        test_info = []
        cnt = 0
        periods = []
        for batch_id, data in enumerate(val_reader()):
            image = np.array([x[0].reshape(image_shape) for x in data]).astype(
                "float32"
            )
            label = np.array([x[1] for x in data]).astype("int64")
            label = label.reshape([-1, 1])

            t1 = time.time()
            _, acc1, _ = exe.run(
                infer_program,
                feed={feed_dict[0]: image, feed_dict[1]: label},
                fetch_list=fetch_targets,
            )
            t2 = time.time()
            period = t2 - t1
            periods.append(period)

            test_info.append(np.mean(acc1) * len(data))
            cnt += len(data)

            if (batch_id + 1) % 100 == 0:
                print(f"{batch_id + 1} images,")
                sys.stdout.flush()
            if (batch_id + 1) == iterations:
                break

        throughput = cnt / np.sum(periods)
        latency = np.average(periods)
        acc1 = np.sum(test_info) / cnt
        return (throughput, latency, acc1, feed_dict)


class TestPostTrainingForResnet50ONNXFormat(TestPostTrainingForResnet50):
    def test_post_training_resnet50(self):
        model = "ResNet-50"
        algo = "min_max"
        round_type = "round"
        data_urls = [
            'http://paddle-inference-dist.bj.bcebos.com/int8/resnet50_int8_model_combined.tar.gz'
        ]
        data_md5s = ['db212fd4e9edc83381aef4533107e60c']
        quantizable_op_type = ["conv2d", "mul"]
        is_full_quantize = False
        is_use_cache_file = False
        is_optimize_model = False
        diff_threshold = 0.025
        onnx_format = True
        self.run_test(
            model,
            'model.pdmodel',
            'model.pdiparams',
            algo,
            round_type,
            data_urls,
            data_md5s,
            "model",
            quantizable_op_type,
            is_full_quantize,
            is_use_cache_file,
            is_optimize_model,
            diff_threshold,
            onnx_format=onnx_format,
        )


if __name__ == '__main__':
    unittest.main()
