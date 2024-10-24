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
from paddle import base

paddle.enable_static()


class TestAvoidTwiceInitialization(unittest.TestCase):
    def test_avoid_twice_initialization(self):
        cur_program = base.Program()
        cur_block = cur_program.current_block()
        var = cur_block.create_parameter(
            initializer=paddle.nn.initializer.Constant(value=0.01),
            shape=[2, 2],
            dtype='float32',
            name='var_a',
        )
        cur_block.append_op(
            type="broadcast",
            inputs={"x": [var]},
            outputs={"out": [var]},
            attrs={'root': 0, 'ring_id': 0},
        )
        cur_block.append_op(
            type="c_sync_comm_stream",
            inputs={'X': [var]},
            outputs={'Out': [var]},
            attrs={'ring_id': 0},
        )
        var2 = cur_block.create_parameter(
            initializer=paddle.nn.initializer.Constant(value=0.01),
            shape=[2, 2],
            dtype='float32',
            name='var_a',
        )


if __name__ == '__main__':
    unittest.main()
