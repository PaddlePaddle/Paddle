# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import paddle.nn as nn
import numpy as np


class MLPLayer(nn.Layer):
    def __init__(self, input_size, hidden_size, output_size, n):
        super(MLPLayer, self).__init__()
        self.linear_first = nn.Linear(input_size, hidden_size)
        self.linear_mid = nn.Sequential(
            ('l0', nn.Linear(hidden_size, hidden_size))
        )
        for i in range(n - 1):
            self.linear_mid.add_sublayer(
                'l' + str(i + 1), nn.Linear(hidden_size, hidden_size)
            )
        self.linear_last = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear_first(x)
        x = self.linear_mid(x)
        x = self.linear_last(x)
        return x.mean()


class TestMultiTensorAdam(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.input_size = 800
        self.hidden_size = 500
        self.output_size = 700
        self.n = 10

    def get_adam_or_adamw_out(
        self, use_multi_tensor_adam, use_adamw, test_dict, test_fp16
    ):

        paddle.seed(10)
        np.random.seed(10)

        model = MLPLayer(
            self.input_size, self.hidden_size, self.output_size, self.n
        )

        if test_fp16:
            model = paddle.amp.decorate(models=model, level='O2')
            scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
            inp = np.random.random((10, self.input_size)).astype("float16")
            inp = paddle.to_tensor(inp)
        else:
            inp = paddle.uniform([10, self.input_size], dtype="float32")
            out = model(inp)

        beta1 = paddle.to_tensor([0.9], dtype="float32")
        beta2 = paddle.to_tensor([0.99], dtype="float32")

        paramters_dict_list = []

        input_paramters = model.parameters()

        if test_dict:
            i = 0
            for param in model.parameters():
                paramters_dict_list.append(
                    {
                        'params': param,
                        'weight_decay': 0.001 * i,
                        'learning_rate': 0.01 * i,
                        'beta1': 0.01 * i,
                    }
                )
                i = i + 1

                input_paramters = paramters_dict_list

        if use_multi_tensor_adam:
            if not use_adamw:
                opt = paddle.incubate.optimizer.MultiTensorAdam(
                    learning_rate=0.1,
                    parameters=input_paramters,
                    weight_decay=0.01,
                    beta1=beta1,
                    beta2=beta2,
                )
            else:
                opt = paddle.incubate.optimizer.MultiTensorAdamW(
                    learning_rate=0.1,
                    parameters=input_paramters,
                    weight_decay=0.01,
                    beta1=beta1,
                    beta2=beta2,
                )
        else:
            if not use_adamw:
                opt = paddle.optimizer.Adam(
                    learning_rate=0.1,
                    parameters=input_paramters,
                    weight_decay=0.01,
                    beta1=beta1,
                    beta2=beta2,
                )
            else:
                opt = paddle.optimizer.AdamW(
                    learning_rate=0.1,
                    parameters=input_paramters,
                    weight_decay=0.01,
                    beta1=beta1,
                    beta2=beta2,
                )

        if not test_fp16:
            out.backward()
            opt.step()
            opt.clear_grad()
        else:
            with paddle.amp.auto_cast(level='O2'):
                out = model(inp)
                loss = paddle.mean(out)
            scaled = scaler.scale(loss)
            scaled.backward()
            scaler.step(opt)
            opt.clear_grad()

        return model.parameters()

    def run_adam_or_adamw(self, use_adamw, test_dict, test_fp16):
        use_multi_tensor_adam = True
        parameters = self.get_adam_or_adamw_out(
            use_multi_tensor_adam, use_adamw, test_dict, test_fp16
        )
        parameters_1 = self.get_adam_or_adamw_out(
            not use_multi_tensor_adam, use_adamw, test_dict, test_fp16
        )
        for i, j in zip(parameters, parameters_1):
            np.testing.assert_allclose(i.numpy(), j.numpy(), atol=1e-5)

    def test_main(self):
        old_device = paddle.get_device()
        for use_gpu in [False, True]:
            if use_gpu and not paddle.is_compiled_with_cuda():
                continue
            if use_gpu:
                paddle.set_device("gpu")
            else:
                paddle.set_device("cpu")
            for use_adamw in [True, False]:
                for test_dict in [True, False]:
                    test_fp16 = False
                    self.run_adam_or_adamw(use_adamw, test_dict, test_fp16)
                    if use_gpu:
                        test_fp16 = True
                        self.run_adam_or_adamw(use_adamw, test_dict, test_fp16)
        paddle.set_device(old_device)


if __name__ == "__main__":
    unittest.main()
