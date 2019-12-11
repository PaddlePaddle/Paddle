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

import unittest
import paddle.fluid as fluid
from paddle.fluid.framework import Program, program_guard


class TestShellOp(unittest.TestCase):
    def _shell_op_(self, params_dict=None, cmd_format='ls /', create_var=True):
        scope = fluid.core.Scope()
        program = Program()
        with fluid.scope_guard(scope):
            with program_guard(program, startup_program=Program()):
                place = fluid.CPUPlace()
                params = []
                if params_dict and type(params_dict) == dict:
                    for k, v in params_dict.items():
                        if create_var:
                            param = scope.var(k)
                            if v:
                                param.set_string(v)
                        params.append(k)

                program.global_block().append_op(
                    type="shell",
                    inputs={},
                    outputs={},
                    attrs={"cmd_format": cmd_format,
                           "cmd_params": params})

            exe = fluid.Executor(place)
            exe.run(program)

    def test_shell_op(self):
        #default no params
        self._shell_op_()
        # exception 1, command format have more placeholder {}
        self.assertRaises(Exception, self._shell_op_, None, 'ls {}')

        # exception 2, param variable doesnt have value
        params_dict = dict()
        params_dict['params'] = None
        self.assertRaises(Exception, self._shell_op_, params_dict, 'ls {}')

        # exception 3, param variable doesnt exists
        self.assertRaises(Exception, self._shell_op_, params_dict, 'ls {}',
                          False)

        # exception 4, command format don't have enouth placeholder {}
        params_dict['params'] = "./"
        self.assertRaises(Exception, self._shell_op_, params_dict, 'ls')

        # correct use
        self._shell_op_(params_dict, 'ls {}')


if __name__ == '__main__':
    unittest.main()
