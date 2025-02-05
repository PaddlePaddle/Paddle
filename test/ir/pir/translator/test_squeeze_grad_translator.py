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


class TestSqueezeGradOpTranslator(test_op_translator.TestOpTranslator):
    def append_op(self):
        self.op_type = "squeeze_grad"
        x = paddle.ones(shape=(100, 1, 1), dtype='float32')
        x.stop_gradient = False
        out = paddle.squeeze(x, [1, -1])
        # To trigger generating squeeze_grad operator in program.
        x_grad = paddle.static.gradients([out], x)

    def test_translator(self):
        self.check()


if __name__ == "__main__":
    unittest.main()
