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

import numpy as np
import random
import paddle

from paddle.fluid.tests.unittests.ipu.op_test_ipu import IPUOpTest
from paddle.inference import Config
from paddle.inference import create_predictor

model_zoo = {
    'ernie': ["/paddle_build/third_party/inference_demo/Ernie/model"],
    'resnet50': [
        "/paddle_build/third_party/inference_demo/resnet50/model/model",
        "/paddle_build/third_party/inference_demo/resnet50/model/params"
    ],
}


class IPUInferenceTest(IPUOpTest):

    @classmethod
    def setUpClass(cls):
        # Get random seeds
        cls._np_rand_state = np.random.get_state()
        cls._py_rand_state = random.getstate()

        cls.SEED = 2021
        np.random.seed(cls.SEED)
        random.seed(cls.SEED)
        paddle.seed(cls.SEED)

    def set_threshold(self,
                      mae_fp32=1e-4,
                      mse_fp32=1e-6,
                      mae_fp16=1e-2,
                      mse_fp16=1e-3):
        self.mae_fp32 = mae_fp32
        self.mse_fp32 = mse_fp32
        self.mae_fp16 = mae_fp16
        self.mse_fp16 = mse_fp16

    def set_fp16(self, enable_fp16):
        self.fp16_mode = enable_fp16

    def set_model(self, model_name):
        self.assertTrue(model_name in model_zoo.keys())
        self.model = model_zoo[model_name]

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def create_predictor(self, exec_mode):
        if len(self.model) == 1:
            config = Config(self.model[0])
        else:
            config = Config(self.model[0], self.model[1])
        config.switch_ir_optim(False)
        config.disable_glog_info()

        if exec_mode == IPUInferenceTest.ExecutionMode.IPU_FP32:
            config.enable_ipu(1, self.batch_size)
        elif exec_mode == IPUInferenceTest.ExecutionMode.IPU_FP16:
            config.enable_ipu(1, self.batch_size)
            config.set_ipu_config(True)

        self.predictor = create_predictor(config)

    def run_model(self, exec_mode):
        # copy img data to input tensor
        input_names = self.predictor.get_input_names()
        for i, name in enumerate(input_names):
            input_tensor = self.predictor.get_input_handle(name)
            input_tensor.reshape(self.data_shape[i])
            if exec_mode == 3:
                input_tensor.copy_from_cpu(self.feed_fp16[i].copy())
            else:
                input_tensor.copy_from_cpu(self.feed_fp32[i].copy())

        # do the inference
        self.predictor.run()

        result = []
        # get out data from output tensor
        output_names = self.predictor.get_output_names()
        for i, name in enumerate(output_names):
            output_tensor = self.predictor.get_output_handle(name)
            output_data = output_tensor.copy_to_cpu()
            result.append(output_data)

        if isinstance(result, list) and len(result) == 1:
            self.output_dict[exec_mode] = result[0]
        else:
            self.output_dict[exec_mode] = result

    def check(self, check_shape=False, output_dict=None):
        if output_dict is None:
            output_dict = self.output_dict
        if len(output_dict) == 0:
            raise ValueError("output_dict is empty")
        cpu_fp32 = output_dict[IPUInferenceTest.ExecutionMode.CPU_FP32]
        ipu_fp32 = output_dict[IPUInferenceTest.ExecutionMode.IPU_FP32]
        cpu_fp32 = np.asarray(cpu_fp32).astype(np.float32).flatten()
        ipu_fp32 = np.asarray(ipu_fp32).astype(np.float32).flatten()

        mae = sum([abs(x - y)
                   for x, y in zip(cpu_fp32, ipu_fp32)]) / len(cpu_fp32)
        mse = sum([(x - y)**2
                   for x, y in zip(cpu_fp32, ipu_fp32)]) / len(cpu_fp32)
        if mae > self.mae_fp32 or mse > self.mse_fp32:
            raise AssertionError(
                f"ipu_fp32 check failed. MAE is {mae}, MSE is {mse}")
        if check_shape:
            self.assertTrue(cpu_fp32.shape == ipu_fp32.shape)

        if self.fp16_mode:
            ipu_fp16 = output_dict[IPUInferenceTest.ExecutionMode.IPU_FP16]
            ipu_fp16 = np.asarray(ipu_fp16).astype(np.float32).flatten()

            mae = sum([abs(x - y)
                       for x, y in zip(cpu_fp32, ipu_fp16)]) / len(cpu_fp32)
            mse = sum([(x - y)**2
                       for x, y in zip(cpu_fp32, ipu_fp16)]) / len(cpu_fp32)
            if mae > self.mae_fp16 or mse > self.mse_fp16:
                raise AssertionError(
                    f"ipu_fp16 check failed. MAE is {mae}, MSE is {mse}")
            if check_shape:
                self.assertTrue(ipu_fp16.shape == cpu_fp32.shape)
