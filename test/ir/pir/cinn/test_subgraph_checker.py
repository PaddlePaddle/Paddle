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
from paddle.decomposition import decompose
from paddle.framework import core


class TestChecker(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()

    def create_program(self, enable_prim=False):
        out_name = 'pt_output_0'
        if enable_prim:
            core._set_prim_forward_enabled(True)
        else:
            core._set_prim_all_enabled(False)

        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            x = paddle.static.data(shape=[16, 4], name='pt_input_0')
            out = paddle.nn.functional.softmax(x)
            fetch_out = paddle._pir_ops.fetch(out, out_name, 0)
            fetch_out.persistable = True
            decompose(main_program, [fetch_out])
        return main_program

    def test_check(self):
        orig_program = self.create_program(enable_prim=False)
        prim_program = self.create_program(enable_prim=True)
        checker = paddle.base.libpaddle.test.SubGraphChecker(
            orig_program, prim_program
        )
        checker.check_result()
        checker.check_speed()


if __name__ == "__main__":
    unittest.main()
