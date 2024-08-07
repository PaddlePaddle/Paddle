#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


class TestBlockRenameVar(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        self.program = paddle.static.Program()
        self.block = self.program.current_block()
        self.var = self.block.create_var(
            name="X", shape=[-1, 23, 48], dtype='float32'
        )
        self.op = self.block.append_op(
            type="abs", inputs={"X": [self.var]}, outputs={"Out": [self.var]}
        )
        self.new_var_name = self.get_new_var_name()

    def get_new_var_name(self):
        return "Y"

    def test_rename_var(self):
        self.block._rename_var(self.var.name, self.new_var_name)
        new_var_name_str = (
            self.new_var_name
            if isinstance(self.new_var_name, str)
            else self.new_var_name.decode()
        )
        self.assertTrue(new_var_name_str in self.block.vars)


class TestBlockRenameVarStrCase2(TestBlockRenameVar):
    def get_new_var_name(self):
        return "ABC"


class TestBlockRenameVarBytes(TestBlockRenameVar):
    def get_new_var_name(self):
        return b"Y"


if __name__ == "__main__":
    unittest.main()
