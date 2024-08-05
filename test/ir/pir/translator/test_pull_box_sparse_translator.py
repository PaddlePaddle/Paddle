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
from paddle.base.layer_helper import LayerHelper

paddle.pir_utils._switch_to_old_ir_()


class TestPullBoxSparseOpTranslator(
    test_op_translator.TestOpWithBackwardTranslator
):
    def append_op(self):
        self.forward_op_type = "pull_box_sparse"
        self.backward_op_type = "push_box_sparse"
        ids = paddle.ones(shape=(1, 1), dtype='float32')
        w = paddle.ones(shape=(1, 1), dtype='float32')
        out = paddle.ones(shape=(1, 1), dtype='float32')
        attrs = {
            'is_sparse': False,
            'is_distributed': False,
            'size': 1,
        }
        forward_helper = LayerHelper(self.forward_op_type)
        forward_helper.append_op(
            type=self.forward_op_type,
            inputs={"W": w, "Ids": [ids]},
            outputs={"Out": [out]},
            attrs=attrs,
        )
        return out

    def test_translator(self):
        self.check()


if __name__ == "__main__":
    unittest.main()
