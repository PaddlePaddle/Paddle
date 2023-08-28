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

import numpy as np

import paddle
from paddle.fluid import core

paddle.enable_static()


class TestPdInplacePass(unittest.TestCase):
    def test_pd_inplace_pass(self):
        place = core.Place()
        place.set_place(paddle.CPUPlace())
        new_scope = paddle.static.Scope()
        main_program = paddle.static.Program()
        with paddle.static.scope_guard(new_scope):
            with paddle.static.program_guard(main_program):
                x = paddle.ones([2, 2], dtype='float32')
                y = paddle.nn.functional.relu(x)

                op_names = [op.name() for op in main_program.block().ops]
                self.assertTrue('pd.relu' in op_names)
                self.assertTrue('pd.relu_' not in op_names)

                exe = paddle.static.Executor()
                (sum_value,) = exe.run(
                    fetch_list=[main_program.block().ops[1].result(0)]
                )
                self.assertEqual(
                    (sum_value == np.ones([2, 2], dtype="float32")).all(), True
                )

                op_names = [op.name() for op in main_program.block().ops]
                self.assertTrue('pd.relu' not in op_names)
                self.assertTrue('pd.relu_' in op_names)


if __name__ == "__main__":
    unittest.main()
