#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid.core as core
import paddle.fluid as fluid


class TestCCommInitAllOp(unittest.TestCase):

    def setUp(self):
        self.place = fluid.CUDAPlace(0)
        self.exe = fluid.Executor(self.place)

    def test_default_attrs(self):
        program = fluid.Program()
        block = program.global_block()
        block.append_op(type='c_comm_init_all', attrs={'ring_id': 0})
        self.exe.run(program)

    def test_init_with_same_ring_id(self):
        program = fluid.Program()
        block = program.global_block()
        block.append_op(type='c_comm_init_all', attrs={'ring_id': 0})
        with self.assertRaises(ValueError):
            self.exe.run(program)

    def test_specifying_devices(self):
        program = fluid.Program()
        block = program.global_block()
        block.append_op(type='c_comm_init_all',
                        attrs={
                            'devices': [0],
                            'ring_id': 1
                        })
        self.exe.run(program)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
