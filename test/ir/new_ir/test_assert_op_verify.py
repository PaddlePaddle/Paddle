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
from paddle.fluid import core
from paddle.static.nn.control_flow import Assert

paddle.enable_static()


class TestOpVerify(unittest.TestCase):
    def test_program(self):
        main_program, start_program = (
            paddle.static.Program(),
            paddle.static.Program(),
        )
        with paddle.static.program_guard(main_program, start_program):
            x = paddle.tensor.fill_constant(
                shape=[2, 3], dtype='float32', value=2.0
            )
            condition = paddle.max(x) < 1.0
            Assert(condition, [x], 10, name="test")
        newir_program = core.translate_newirprogram(main_program.desc)
        newir_program.print()
        self.assertTrue(newir_program is not None)


if __name__ == "__main__":
    unittest.main()
