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

from test_case_base import (
    TestCaseBase,
    test_instruction_translator_cache_context,
)

import paddle
from paddle.vision.models.resnet import resnet18


def resnet_call(x: paddle.Tensor, net: paddle.nn.Layer):
    return net(x)


class TestResNet(TestCaseBase):
    def test_resnet_eval(self):
        x = paddle.rand((10, 3, 224, 224))
        net = resnet18(pretrained=False)
        net.eval()
        with test_instruction_translator_cache_context() as ctx:
            self.assert_results(resnet_call, x, net)
            self.assertEqual(ctx.translate_count, 1)
            self.assert_results(resnet_call, x, net)  # cache hit
            self.assertEqual(ctx.translate_count, 1)
            net.train()
            self.assert_results(resnet_call, x, net)  # cache miss
            self.assertEqual(ctx.translate_count, 2)

    def test_resnet_train(self):
        x = paddle.rand((10, 3, 224, 224))
        net = resnet18(pretrained=False)
        net.train()
        with test_instruction_translator_cache_context() as ctx:
            self.assert_results(resnet_call, x, net)
            self.assertEqual(ctx.translate_count, 1)
            self.assert_results(resnet_call, x, net)  # cache hit
            self.assertEqual(ctx.translate_count, 1)
            net.eval()
            self.assert_results(resnet_call, x, net)  # cache miss
            self.assertEqual(ctx.translate_count, 2)


if __name__ == "__main__":
    unittest.main()
