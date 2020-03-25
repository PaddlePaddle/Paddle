# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
from op_test import OpTest
import paddle.tensor as tensor
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard


class TestMulOpError(unittest.TestCase):
    def test_nonzero_api(self):
        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            x = fluid.layers.data(name='x', shape=[-1, 2])
            y_tuple = tensor.nonzero(x, as_tuple=True)
            self.assertEqual(len(y_tuple), 2)
            y = tensor.nonzero(x, as_tuple=False)
            z = fluid.layers.concat(list(y_tuple), axis=1)
        data = np.array([[True, False], [False, True]])
        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(startup_program)
        outs = exe.run(main_program,
                       feed={'x': data},
                       fetch_list=[z.name],
                       return_numpy=False)
        expect_out = np.array([[0, 0], [1, 1]])
        self.assertTrue(np.allclose(expect_out, np.array(outs[0])))


if __name__ == "__main__":
    unittest.main()
