#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import paddle.fluid as fluid
from paddle.fluid.framework import Program, program_guard

class TestShellOp(unittest.TestCase):
    def test_shell_op_no_params():
        scope = fluid.core.Scope()
        program = Program()
        with fluid.scope_guard(scope):
            with program_guard(program, startup_program=Program()):
                place = fluid.CPUPlace()
                program.global_block().append_op(
                    type="shell",
                    inputs={},
                    outputs={},
                    attrs={
                        "cmd_format": 'ls /',
                    })

            exe = fluid.Executor(place)
            exe.run(program)

if __name__ == '__main__':
    unittest.main()

