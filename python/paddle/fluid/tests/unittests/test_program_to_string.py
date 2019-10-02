# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import unittest


class TestProgram(unittest.TestCase):
    def test_program_to_string(self):
        prog = fluid.default_main_program()
        a = fluid.layers.data(
            name="X", shape=[2, 3], dtype="float32", append_batch_size=False)
        c = fluid.layers.fc(a, size=3)
        prog_string = prog.to_string(throw_on_error=True, with_details=False)
        prog_string_with_details = prog.to_string(
            throw_on_error=False, with_details=True)
        assert prog_string is not None
        assert len(prog_string_with_details) > len(prog_string)


if __name__ == '__main__':
    unittest.main()
