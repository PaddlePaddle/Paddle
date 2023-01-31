# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import static

paddle.enable_static()


class TestCrossStepOverlap(unittest.TestCase):
    def setUp(self):
        self.shape = [16, 513, 513, 19]
        self.x_value = 2
        self.y_value = 3
        self.overlap_op_num = 1500
        self.step_num = 3

    def test_cross_step_overlap(self):
        if not paddle.fluid.core.is_compiled_with_cuda():
            return

        # In this test case, z=x+y is calculated in the default stream,
        # and at the same time, numerous reduce_min ops that output to y
        # are executed in another stream (i.e., the custom stream).
        # These reduce_min ops are carefully designed that their kernel
        # calculation will overlap with the fill_constant kernels (output
        # to x and y) in the next step, and therefore cross-step multi-stream
        # synchronization is required. An Event should be recorded after the
        # last reduce_min in the first step and waited before the fill_constant
        # in the second step. Otherwise, the result of z will be wrong.
        program = static.Program()
        with static.program_guard(program):
            x = paddle.full(
                self.shape, fill_value=self.x_value, dtype='float64'
            )
            y = paddle.full(
                self.shape, fill_value=self.y_value, dtype='float64'
            )
            z = paddle.add(x, y)

            block = program.global_block()
            block.var(x.name).desc.set_persistable(True)
            block.var(y.name).desc.set_persistable(True)
            for i in range(self.overlap_op_num):
                block.append_op(
                    type='reduce_min',
                    inputs={'X': x.name},
                    outputs={'Out': y.name},
                    attrs={'axis': 0, 'keepdim': True},
                )
                block.ops[-1].dist_attr.execution_stream = "custom"

            exe = static.Executor()
            results = []
            for i in range(self.step_num):
                result = exe.run(program, fetch_list=[z])
                results.append(result)

            for result in results:
                self.assertAlmostEqual(
                    np.sum(result),
                    (self.x_value + self.y_value) * np.prod(self.shape),
                )


if __name__ == "__main__":
    unittest.main()
