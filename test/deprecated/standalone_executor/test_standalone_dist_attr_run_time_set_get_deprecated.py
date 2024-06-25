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
from paddle.static import Program, program_guard

paddle.enable_static()


class TestOperatorDistAttrSetGet(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def _build_startup_program_and_train_program(self):
        startup_program = Program()
        train_program = Program()
        with program_guard(train_program, startup_program):
            data = paddle.static.data(
                name='X', shape=[1024, 1], dtype='float32'
            )
            hidden = paddle.static.nn.fc(data, 10)
            loss = paddle.mean(hidden)
            paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)
        return startup_program, train_program, loss

    def test_run_time_us_set_get_method(self):
        '''
        * test if the newly added "run_time_us_" actually works (set then get)
        '''
        (
            startup_program,
            train_program,
            loss,
        ) = self._build_startup_program_and_train_program()
        global_block = startup_program.global_block()
        global_block.ops[0].dist_attr.run_time_us = 1.0  # set
        dt = global_block.ops[0].dist_attr.run_time_us  # get
        self.assertTrue(dt == 1.0)


if __name__ == "__main__":
    unittest.main()
