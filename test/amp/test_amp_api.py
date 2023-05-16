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

from amp_base_models import AmpTestBase

import paddle


class TestAutoCast(AmpTestBase):
    def setUp(self):
        self._conv = paddle.nn.Conv2D(
            in_channels=1, out_channels=6, kernel_size=3, bias_attr=False
        )
        self._linear = paddle.nn.Linear(in_features=4, out_features=4)

    def test_amp_OD_level(self):
        with paddle.amp.auto_cast(level='OD'):
            out1 = self._conv(paddle.rand(shape=[1, 1, 6, 6], dtype='float32'))
            out2 = out1 + paddle.rand(shape=out1.shape, dtype='float16')
            out3 = self._linear(out2)

        self.assertEqual(out1.dtype, paddle.float16)
        self.assertEqual(out2.dtype, paddle.float32)
        self.assertEqual(out3.dtype, paddle.float32)


class TestGradScaler(AmpTestBase):
    def test_amp_grad_scaler(self):
        model = paddle.nn.Conv2D(3, 2, 3)
        optimizer = paddle.optimizer.SGD(
            learning_rate=0.01, parameters=model.parameters()
        )
        scaler = paddle.amp.GradScaler()
        data = paddle.rand([1, 3, 8, 8], dtype='float32')
        paddle.amp.debugging.enable_operator_stats_collection()
        with paddle.amp.auto_cast(
            custom_black_list=['conv2d'], dtype='bfloat16'
        ):
            out = model(data)
            loss = out.mean()
        scaled = scaler.scale(loss)
        scaled.backward()
        scaler.minimize(optimizer, scaled)
        optimizer.clear_grad()
        paddle.amp.debugging.disable_operator_stats_collection()
        op_list = paddle.fluid.core.get_low_precision_op_list()

        self.assertEqual(scaler._enable, False)
        self.assertEqual(scaler._use_dynamic_loss_scaling, False)
        self.assertTrue('scale' not in op_list)
        self.assertTrue('check_finite_and_unscale' not in op_list)


if __name__ == '__main__':
    unittest.main()
