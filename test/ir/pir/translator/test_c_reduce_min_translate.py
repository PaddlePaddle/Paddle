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

import paddle
from paddle import pir
from paddle.base import core
from paddle.base.layer_helper import LayerHelper

paddle.enable_static()


class TestCReduceMinOpTranscriber(unittest.TestCase):
    def test_program(self):
        place = core.Place()
        place.set_place(paddle.CPUPlace())

        new_scope = paddle.static.Scope()
        main_program = paddle.static.Program()
        with paddle.static.scope_guard(new_scope):
            with paddle.static.program_guard(main_program):
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
        l = pir.translate_to_pir(main_program.desc)
        assert (
            l.global_block().ops[2].name() == "pd_op.c_reduce_min"
        ), "c_reduce_min should be translated to c_reduce_min"


if __name__ == "__main__":
    unittest.main()
