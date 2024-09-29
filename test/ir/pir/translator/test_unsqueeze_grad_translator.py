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

import test_op_translator

import paddle

paddle.pir_utils._switch_to_old_ir_()


class TestUnsqueezeGradOpTranslator(test_op_translator.TestOpTranslator):
    def append_op(self):
        self.op_type = "unsqueeze_grad"
        x = paddle.ones(shape=(100, 2), dtype='float32')
        x.stop_gradient = False
        out = paddle.unsqueeze(x, [1, -1])
        # To trigger generating unsqueeze_grad operator in program.
        x_grad = paddle.static.gradients([out], x)

    def test_translator(self):
        self.check()


class TestUnsqueezeGradOpWithAxesTensor(TestUnsqueezeGradOpTranslator):
    def append_op(self):
        self.op_type = "unsqueeze_grad"
        x = paddle.ones(shape=(100, 2), dtype='float32')
        x.stop_gradient = False
        axes = paddle.to_tensor([1, -1], dtype='int32')
        out = paddle.unsqueeze(x, axes)
        # To trigger generating unsqueeze_grad operator in program.
        x_grad = paddle.static.gradients([out], x)


class TestUnsqueezeGradOpWithAxesListTensor(TestUnsqueezeGradOpTranslator):
    def append_op(self):
        self.op_type = "unsqueeze_grad"
        x = paddle.ones(shape=(100, 2), dtype='float32')
        x.stop_gradient = False
        axes = [
            paddle.to_tensor([1], dtype='int32'),
            paddle.to_tensor([-1], dtype='int32'),
        ]
        out = paddle.unsqueeze(x, axes)
        # To trigger generating unsqueeze_grad operator in program.
        x_grad = paddle.static.gradients([out], x)


if __name__ == "__main__":
    unittest.main()
