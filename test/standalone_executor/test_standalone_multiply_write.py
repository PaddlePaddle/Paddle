# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from test_standalone_controlflow import TestCompatibility

import paddle

# from paddle.base.framework import Program

paddle.enable_static()


class TestMultiplyWrite(TestCompatibility):
    def _get_feed(self):
        """return the feeds"""
        return None

    def build_program(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            out = paddle.full((1,), 1)
            inp1 = paddle.full((1,), 2)
            inp2 = paddle.full((1,), 3)

            paddle.assign(inp1, out)
            paddle.assign(inp2, out)
        return main_program, startup_program, out

    def run_dygraph_once(self, feed):
        out = paddle.full((1,), 1)
        inp1 = paddle.full((1,), 2)
        inp2 = paddle.full((1,), 3)
        paddle.assign(inp1, out)
        paddle.assign(inp2, out)
        return [out.numpy()]

    def setUp(self):
        self.place = paddle.CPUPlace()
        self.iter_run = 5


if __name__ == "__main__":
    unittest.main()
