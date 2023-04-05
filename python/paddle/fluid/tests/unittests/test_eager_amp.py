# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

import paddle
import paddle.fluid.core as core


class SimpleNet(paddle.nn.Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = paddle.nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.linear(x)
        return x


@unittest.skipIf(
    not core.is_float16_supported(core.CUDAPlace(0)), "core is not support fp16"
)
class TestMasterGrad(unittest.TestCase):
    def check_results(self, fp16_grads, fp32_grads):
        for i in range(len(fp16_grads)):
            self.assertEqual(fp32_grads[i].dtype, paddle.float32)
            np.testing.assert_allclose(fp16_grads[i], fp32_grads[i])

    def run_dygraph(self):
        accumulate_batchs_num = 2
        total_steps = 4
        model = SimpleNet(2, 4)
        opt = paddle.optimizer.AdamW(parameters=model.parameters())
        model, opt = paddle.amp.decorate(
            model, optimizers=opt, level='O2', master_grad=True
        )
        scaler = paddle.amp.GradScaler()

        for i in range(total_steps):
            x = np.random.random((2, 2)).astype('float32')
            label = np.random.random((2, 4)).astype('float32')
            with paddle.amp.auto_cast(level='O2'):
                out = model(paddle.to_tensor(x))
                loss = paddle.nn.functional.l1_loss(
                    out, paddle.to_tensor(label)
                )
            scaled = scaler.scale(loss)
            scaled.backward()
            fp16_grads = [model.linear.weight.grad, model.linear.bias.grad]
            fp32_grads = [model.linear.weight.grad, model.linear.bias.grad]
            self.check_results(fp16_grads, fp32_grads)
            if (i + 1) % accumulate_batchs_num == 0:
                scaler.step(opt)
                scaler.update()
                opt.clear_grad()

    def test_master_grad(self):
        self.run_dygraph()


if __name__ == '__main__':
    unittest.main()
