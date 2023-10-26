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
from paddle.base import core


class SimpleNet(paddle.nn.Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = paddle.nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.linear(x)
        return x


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_float16_supported(core.CUDAPlace(0)),
    "core is not complied with CUDA and not support the float16",
)
@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or paddle.device.cuda.get_device_capability()[0] < 7.0,
    "run test when gpu's compute capability is at least 7.0.",
)
class TestMasterGrad(unittest.TestCase):
    def check_results(
        self, fp32_grads, op_list, total_steps, accumulate_batchs_num
    ):
        for grad in fp32_grads:
            self.assertEqual(grad.dtype, paddle.float32)
        # fp16 calls
        self.assertEqual(int(op_list['matmul_v2'].split(',')[0]), total_steps)
        self.assertEqual(
            int(op_list['adam_'].split(',')[0]),
            2 * (total_steps / accumulate_batchs_num),
        )
        self.assertEqual(
            int(op_list['transfer_dtype'].split(',')[0]),
            total_steps + total_steps * 2,
        )

    def run_dygraph(self, total_steps, accumulate_batchs_num, model, optimizer):
        model, opt = paddle.amp.decorate(
            model, optimizers=optimizer, level='O2', master_grad=True
        )
        scaler = paddle.amp.GradScaler()
        paddle.amp.debugging.enable_operator_stats_collection()
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
            fp32_grads = [model.linear.weight.grad, model.linear.bias.grad]
            if (i + 1) % accumulate_batchs_num == 0:
                scaler.step(opt)
                scaler.update()
                opt.clear_grad()
        paddle.amp.debugging.disable_operator_stats_collection()
        op_list = paddle.base.core.get_low_precision_op_list()
        return fp32_grads, op_list

    def test_adam_master_grad(self):
        total_steps = 4
        accumulate_batchs_num = 2
        model = SimpleNet(2, 4)
        opt = paddle.optimizer.Adam(parameters=model.parameters())
        fp32_grads, op_list = self.run_dygraph(
            total_steps, accumulate_batchs_num, model, opt
        )
        self.check_results(
            fp32_grads, op_list, total_steps, accumulate_batchs_num
        )

    def test_momentum_master_grad(self):
        total_steps = 4
        accumulate_batchs_num = 1
        model = SimpleNet(2, 4)
        L1Decay = paddle.regularizer.L1Decay(0.0001)
        opt = paddle.optimizer.Momentum(
            parameters=model.parameters(), weight_decay=L1Decay
        )
        fp32_grads, op_list = self.run_dygraph(
            total_steps, accumulate_batchs_num, model, opt
        )
        for grad in fp32_grads:
            self.assertEqual(grad.dtype, paddle.float32)


if __name__ == '__main__':
    unittest.main()
