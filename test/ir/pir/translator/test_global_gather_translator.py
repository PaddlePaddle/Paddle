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


class TestGlobalGatherOpTranslator(
    test_op_translator.TestOpWithBackwardTranslator
):
    def append_op(self):
        self.forward_op_type = "global_gather"
        self.backward_op_type = "global_scatter"
        x = paddle.ones(
            shape=(
                1,
                1,
            ),
            dtype='int64',
        )
        local_count = paddle.ones(shape=(1,), dtype='int64')
        global_count = paddle.ones(shape=(1,), dtype='int64')
        x.stop_gradient = False
        local_count.stop_gradient = False
        global_count.stop_gradient = False
        out = paddle.ones(shape=(1,), dtype='int64')
        attrs = {'ring_id': 0, 'use_calc_stream': False}
        helper = LayerHelper(self.forward_op_type)
        helper.append_op(
            type=self.forward_op_type,
            inputs={
                "X": x,
                'local_count': local_count,
                'global_count': global_count,
            },
            outputs={"Out": out},
            attrs=attrs,
        )
        return out

    def test_translator(self):
        self.check()


if __name__ == "__main__":
    unittest.main()
