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

    def get_adam_or_adamw_out(self, use_multi_tesnor_adam, use_adamw):

        paddle.seed(10)

        model = MLPLayer(
            self.input_size, self.hidden_size, self.output_size, self.n
        )

        inp = paddle.uniform([10, self.input_size], dtype="float32")

        out = model(inp)

        beta1 = paddle.to_tensor([0.9], dtype="float32")
        beta2 = paddle.to_tensor([0.99], dtype="float32")

        if use_multi_tesnor_adam:
            opt = paddle.incubate.optimizer.MultiTensorAdam(
                learning_rate=0.1,
                parameters=model.parameters(),
                weight_decay=0.01,
                beta1=beta1,
                beta2=beta2,
                use_adamw=use_adamw,
            )
        else:
            if not use_adamw:
                opt = paddle.optimizer.Adam(
                    learning_rate=0.1,
                    parameters=model.parameters(),
                    weight_decay=0.01,
                    beta1=beta1,
                    beta2=beta2,
                )
            else:
                opt = paddle.optimizer.AdamW(
                    learning_rate=0.1,
                    parameters=model.parameters(),
                    weight_decay=0.01,
                    beta1=beta1,
                    beta2=beta2,
                )
        out.backward()
        opt.step()
        opt.clear_grad()

        return model.parameters()

    def get_adam_or_adamw_dict_out(self, use_multi_tesnor_adam, use_adamw):

        paddle.seed(10)

        model = MLPLayer(
            self.input_size, self.hidden_size, self.output_size, self.n
        )

        inp = paddle.uniform([10, self.input_size], dtype="float32")

        out = model(inp)

        beta1 = paddle.to_tensor([0.9], dtype="float32")
        beta2 = paddle.to_tensor([0.99], dtype="float32")

        paramters_dict_list = []
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

        if use_multi_tesnor_adam:
            opt = paddle.incubate.optimizer.MultiTensorAdam(
                learning_rate=0.1,
                parameters=paramters_dict_list,
                weight_decay=0.01,
                beta1=beta1,
                beta2=beta2,
                use_adamw=use_adamw,
            )
        else:
            if not use_adamw:
                opt = paddle.optimizer.Adam(
                    learning_rate=0.1,
                    parameters=paramters_dict_list,
                    weight_decay=0.01,
                    beta1=beta1,
                    beta2=beta2,
                )
            else:
                opt = paddle.optimizer.AdamW(
                    learning_rate=0.1,
                    parameters=paramters_dict_list,
                    weight_decay=0.01,
                    beta1=beta1,
                    beta2=beta2,
                )
        out.backward()
        opt.step()
        opt.clear_grad()

        return model.parameters()

    def get_adam_or_adamw_fp16_out(self, use_multi_tesnor_adam, use_adamw):

        paddle.seed(10)
        np.random.seed(10)

        model = MLPLayer(
            self.input_size, self.hidden_size, self.output_size, self.n
        )
        model = paddle.amp.decorate(models=model, level='O2')
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

        inp = np.random.random((10, self.input_size)).astype("float16")
        inp = paddle.to_tensor(inp)

        beta1 = paddle.to_tensor([0.9], dtype="float32")
        beta2 = paddle.to_tensor([0.99], dtype="float32")

        if use_multi_tesnor_adam:
            opt = paddle.incubate.optimizer.MultiTensorAdam(
                learning_rate=0.1,
                parameters=model.parameters(),
                weight_decay=0.01,
                beta1=beta1,
                beta2=beta2,
                use_adamw=use_adamw,
            )
        else:
            if not use_adamw:
                opt = paddle.optimizer.Adam(
                    learning_rate=0.1,
                    parameters=model.parameters(),
                    weight_decay=0.01,
                    beta1=beta1,
                    beta2=beta2,
                )
            else:
                opt = paddle.optimizer.AdamW(
                    learning_rate=0.1,
                    parameters=model.parameters(),
                    weight_decay=0.01,
                    beta1=beta1,
                    beta2=beta2,
                )

        with paddle.amp.auto_cast(level='O2'):
            out = model(inp)
            loss = paddle.mean(out)
        scaled = scaler.scale(loss)
        scaled.backward()
        scaler.step(opt)
        opt.clear_grad()

        return model.parameters()

    def get_adam_or_adamw_dict_fp16_out(self, use_multi_tesnor_adam, use_adamw):

        paddle.seed(10)
        np.random.seed(10)

        model = MLPLayer(
            self.input_size, self.hidden_size, self.output_size, self.n
        )
        model = paddle.amp.decorate(models=model, level='O2')
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

        inp = np.random.random((10, self.input_size)).astype("float16")
        inp = paddle.to_tensor(inp)

        beta1 = paddle.to_tensor([0.9], dtype="float32")
        beta2 = paddle.to_tensor([0.99], dtype="float32")

        paramters_dict_list = []
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

        if use_multi_tesnor_adam:
            opt = paddle.incubate.optimizer.MultiTensorAdam(
                learning_rate=0.1,
                parameters=paramters_dict_list,
                weight_decay=0.01,
                beta1=beta1,
                beta2=beta2,
                use_adamw=use_adamw,
            )
        else:
            if not use_adamw:
                opt = paddle.optimizer.Adam(
                    learning_rate=0.1,
                    parameters=paramters_dict_list,
                    weight_decay=0.01,
                    beta1=beta1,
                    beta2=beta2,
                )
            else:
                opt = paddle.optimizer.AdamW(
                    learning_rate=0.1,
                    parameters=paramters_dict_list,
                    weight_decay=0.01,
                    beta1=beta1,
                    beta2=beta2,
                )
        with paddle.amp.auto_cast(level='O2'):
            out = model(inp)
            loss = paddle.mean(out)
        scaled = scaler.scale(loss)
        scaled.backward()
        scaler.step(opt)
        opt.clear_grad()

        return model.parameters()

    def run_adam_or_adamw(self, use_adamw):
        use_multi_tesnor_adam = True
        parameters = self.get_adam_or_adamw_out(
            use_multi_tesnor_adam, use_adamw
        )
        parameters_1 = self.get_adam_or_adamw_out(
            not use_multi_tesnor_adam, use_adamw
        )
        for i, j in zip(parameters, parameters_1):
            self.assertTrue(np.array_equal(i.numpy(), j.numpy()))

    def run_adam_or_adamw_fp16(self, use_adamw):
        use_multi_tesnor_adam = True
        parameters = self.get_adam_or_adamw_fp16_out(
            use_multi_tesnor_adam, use_adamw
        )
        parameters_1 = self.get_adam_or_adamw_fp16_out(
            not use_multi_tesnor_adam, use_adamw
        )
        for i, j in zip(parameters, parameters_1):
            self.assertTrue(np.array_equal(i.numpy(), j.numpy()))

    def run_adam_or_adamw_dict(self, use_adamw):
        use_multi_tesnor_adam = True
        parameters = self.get_adam_or_adamw_dict_out(
            use_multi_tesnor_adam, use_adamw
        )
        parameters_1 = self.get_adam_or_adamw_dict_out(
            not use_multi_tesnor_adam, use_adamw
        )
        for i, j in zip(parameters, parameters_1):
            self.assertTrue(np.array_equal(i.numpy(), j.numpy()))

    def run_adam_or_adamw_dict_fp16(self, use_adamw):
        use_multi_tesnor_adam = True
        parameters = self.get_adam_or_adamw_dict_fp16_out(
            use_multi_tesnor_adam, use_adamw
        )
        parameters_1 = self.get_adam_or_adamw_dict_fp16_out(
            not use_multi_tesnor_adam, use_adamw
        )
        for i, j in zip(parameters, parameters_1):
            self.assertTrue(np.array_equal(i.numpy(), j.numpy()))

    def test_main(self):
        for use_adamw in [True, False]:
            self.run_adam_or_adamw(use_adamw)
            self.run_adam_or_adamw_fp16(use_adamw)
            self.run_adam_or_adamw_dict(use_adamw)
            self.run_adam_or_adamw_dict_fp16(use_adamw)


if __name__ == "__main__":
    unittest.main()
