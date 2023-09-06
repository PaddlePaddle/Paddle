# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import parameterized as param

import paddle
from paddle.base import core
from paddle.incubate.autograd import primapi, primx



class TestToPrim(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        core._set_prim_forward_enabled(True)

    def tearDown(self):
        core._set_prim_forward_enabled(False)
        paddle.disable_static()

    @param.parameterized.expand((({'dropout'},),))
    def test_blacklist(self, blacklist):
        program = paddle.static.Program()
        with paddle.static.program_guard(program):
            paddle.nn.functional.softmax(
                paddle.nn.functional.dropout(paddle.rand((1,)))
            )
        primapi.to_prim(program.blocks, blacklist=blacklist)
        ops = tuple(op.type for op in program.block(0).ops)
        self.assertTrue(all(tuple(op in ops for op in blacklist)))

    @param.parameterized.expand((({'dropout'},),))
    def test_whitelist(self, whitelist):
        program = paddle.static.Program()
        with paddle.static.program_guard(program):
            paddle.nn.functional.softmax(
                paddle.nn.functional.dropout(paddle.rand((1,)))
            )
        primapi.to_prim(program.blocks, whitelist=whitelist)
        ops = tuple(op.type for op in program.block(0).ops)
        self.assertTrue(all(tuple(op not in ops for op in whitelist)))

    @param.parameterized.expand((({'softmax'}, {'softmax', 'dropout'}),))
    def test_both_not_empty(self, blacklist, whitelist):
        program = paddle.static.Program()
        with paddle.static.program_guard(program):
            paddle.nn.functional.softmax(
                paddle.nn.functional.dropout(paddle.rand((1,)))
            )
        primapi.to_prim(
            program.blocks, blacklist=blacklist, whitelist=whitelist
        )
        ops = tuple(op.type for op in program.block(0).ops)
        self.assertTrue(all(tuple(op in ops for op in blacklist)))

    @param.parameterized.expand(((('dropout',), 'softmax'),))
    def test_type_error(self, blacklist, whitelist):
        program = paddle.static.Program()
        with paddle.static.program_guard(program):
            paddle.nn.functional.softmax(
                paddle.nn.functional.dropout(paddle.rand((1,)))
            )
        with self.assertRaises(TypeError):
            primapi.to_prim(
                program.blocks, blacklist=blacklist, whitelist=whitelist
            )


if __name__ == '__main__':
    unittest.main()
