# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np
from inference_pass_test import InferencePassTest
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import PassVersionChecker
from paddle.inference import Config
from paddle.inference import create_predictor
from paddle.inference.contrib import utils


class TestCallback:
    def __init__(self):
        self.num = 1024

    def test(self):
        print(self.num)


class CopyFromTensor(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 128, 768], dtype="float32")
            data_y = fluid.data(name="y", shape=[-1, 128, 768], dtype="float32")
            fc_out1 = fluid.layers.fc(input=data,
                                      size=3072,
                                      num_flatten_dims=2,
                                      act="relu")
            fc_out2 = fluid.layers.fc(input=fc_out1,
                                      size=768,
                                      num_flatten_dims=2)

        self.feeds = {"data": np.random.random((4, 128, 768)).astype("float32")}
        self.fetch_list = [fc_out2]
        self.check_output()
        self.place_1_gpu = False
        self.place_2_gpu = False

    def execute_two_models(self):
        config_1 = Config(self.path)
        if self.place_1_gpu:
            config_1.enable_use_gpu(1000, 0)

        config_2 = Config(self.path)
        if self.place_2_gpu:
            config_2.enable_use_gpu(1000, 0)

        predictor_1 = create_predictor(config_1)
        input_names = predictor_1.get_input_names()
        input_tensor_1 = predictor_1.get_input_handle(input_names[0])
        input_tensor_1.copy_from_cpu(np.array(list(self.feeds.values())[0]))
        self.assertTrue(predictor_1.run())
        output_names = predictor_1.get_output_names()
        output_tensor_1 = predictor_1.get_output_handle(output_names[0])

        predictor_2 = create_predictor(config_2)
        input_tensor_2 = predictor_2.get_input_handle(input_names[0])

        #copy tensor to tensor
        utils.copy_tensor(input_tensor_2, output_tensor_1)
        self.assertTrue(predictor_2.run())

        output_tensor_2 = predictor_2.get_output_handle(output_names[0])
        output_data = output_tensor_2.copy_to_cpu()

    def test_cpu_to_cpu(self):
        self.place_1_gpu = False
        self.place_2_gpu = False
        self.execute_two_models()

    def test_cpu_to_gpu(self):
        self.place_1_gpu = False
        self.place_2_gpu = True
        self.execute_two_models()


if __name__ == "__main__":
    unittest.main()
