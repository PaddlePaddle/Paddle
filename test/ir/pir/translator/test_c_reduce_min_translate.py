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

from test_op_transcriber import TestOpTranscriber

import paddle
from paddle.base.layer_helper import LayerHelper


class TestCReduceMinOpTranscriber(TestOpTranscriber):
    def build_model(self):
        with paddle.static.scope_guard(self.new_scope):
            with paddle.static.program_guard(self.main_program):
                x = paddle.ones(shape=(100, 2, 3), dtype='float32')
                y = paddle.ones(shape=(100, 2, 3), dtype='float32')
                attrs = {'ring_id': 0, 'root_id': 0, 'use_calc_stream': False}
                helper = LayerHelper('c_reduce_min')
                helper.append_op(
                    type="c_reduce_min",
                    inputs={"X": x},
                    outputs={"Out": y},
                    attrs=attrs,
                )

    def test_translator(self):
        self.op_name = "c_reduce_min"
        self.check()


if __name__ == "__main__":
    unittest.main()
