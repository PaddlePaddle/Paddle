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
"""Test cloud role maker."""

import unittest

import numpy as np

import paddle
from paddle import base
from paddle.base import core


class TestGeneratorSeed(unittest.TestCase):
    #     """
    #     Test cases for cpu generator seed.
    #     """
    def test_gen_TruncatedNormal_initializer(self):
        base.disable_dygraph()

        gen = paddle.seed(123123143)
        cur_state = gen.get_state()

        startup_program = base.Program()
        train_program = base.Program()
        with base.program_guard(train_program, startup_program):
            # example 1:
            # attr shape is a list which doesn't contain tensor Variable.
            x = paddle.uniform(shape=[2, 10])
            result_1 = paddle.static.nn.fc(
                x,
                size=10,
                weight_attr=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0, std=2.0
                ),
            )
            result_2 = paddle.static.nn.fc(
                x,
                size=10,
                weight_attr=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0, std=2.0
                ),
            )

            exe = base.Executor(base.CPUPlace())
            exe.run(startup_program)
            out1 = exe.run(
                train_program, feed={}, fetch_list=[result_1, result_2]
            )

        gen.manual_seed(123123143)
        with base.program_guard(train_program, startup_program):
            exe.run(startup_program)
            out2 = exe.run(
                train_program, feed={}, fetch_list=[result_1, result_2]
            )

        out1_res1 = np.array(out1[0])
        out1_res2 = np.array(out1[1])
        out2_res1 = np.array(out2[0])
        out2_res2 = np.array(out2[1])

        if not core.is_compiled_with_cuda():
            print(">>>>>>> sampling id static >>>>>>>")
            np.testing.assert_allclose(out1_res1, out2_res1, rtol=1e-05)
            np.testing.assert_allclose(out1_res2, out2_res2, rtol=1e-05)
            self.assertTrue(not np.allclose(out1_res2, out1_res1))


if __name__ == "__main__":
    unittest.main()
