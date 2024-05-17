# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.distributed.auto_parallel.static.helper import ProgramHelper


class SimpleLayer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fc = paddle.nn.Linear(10, 10)

    def forward(self, x, target):
        out = self.fc(x)
        loss = paddle.nn.functional.mse_loss(out, target)
        return loss


class TestProgramHelper(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.layer = SimpleLayer()
        self.loss_func = None
        self.metrics = None
        self.input_spec = [
            paddle.static.InputSpec(shape=[None, 10], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 10], dtype='float32'),
        ]
        self.label_spec = None
        self.helper = ProgramHelper(
            self.layer,
            self.loss_func,
            self.metrics,
            self.input_spec,
            self.label_spec,
        )
        paddle.enable_static()

    def test_train(self):
        self.helper.build_program('train')
        paddle.static.gradients(self.helper.loss_vars, self.helper.input_vars)
        self.check_ops(['matmul_v2_grad', 'square_grad', 'reduce_mean_grad'])

    def test_eval(self):
        self.helper.build_program('eval')
        self.check_ops(['matmul_v2', 'square', 'reduce_mean'])

    def test_predict(self):
        self.helper.build_program('predict')
        self.check_ops(['matmul_v2', 'square', 'reduce_mean'])

    def test_optimizer(self):
        self.helper.build_program('train')
        paddle.static.gradients(self.helper.loss_vars, self.helper.input_vars)
        optimizer = paddle.optimizer.Adam(
            learning_rate=0.1, parameters=self.layer.parameters()
        )
        self.helper.apply_optimizer(optimizer)
        self.check_ops(['adam'])

    def check_ops(self, target_ops):
        all_ops = [op.type for op in self.helper.main_program.blocks[0].ops]
        for op in target_ops:
            self.assertIn(op, all_ops)


if __name__ == '__main__':
    unittest.main()
