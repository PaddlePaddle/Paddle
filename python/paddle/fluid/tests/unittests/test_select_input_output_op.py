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
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.layers as layers
from paddle.fluid.backward import append_backward
from paddle.fluid.executor import Executor
from paddle.fluid.framework import Program, program_guard
from paddle.fluid.layers.control_flow import select_input, select_output


class TestSplitMergeSelectedVarOps(unittest.TestCase):
    def test_forward_backward_list_output(self):
        for branch_num in range(2, 10):
            program = Program()
            with program_guard(program):
                x = layers.data(name='x', shape=[2], dtype='float32')
                x.stop_gradient = False  # For test gradient
                mask = layers.data(name='mask', shape=[1], dtype='int32')

                outputs = []
                for i in range(branch_num):
                    out = program.current_block().create_var(
                        dtype='float32', type=core.VarDesc.VarType.LOD_TENSOR)
                    outputs.append(out)

                select_output(x, outputs, mask)
                y = select_input(outputs, mask)
                mean = layers.mean(y)
                append_backward(mean)

            place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
            ) else fluid.CPUPlace()
            exe = Executor(place)

            feed_x = np.asarray([1.3, -1.4]).astype(np.float32)
            for i in range(branch_num):
                feed_mask = np.asarray([i]).astype(np.int32)
                ret = exe.run(program,
                              feed={'x': feed_x,
                                    'mask': feed_mask},
                              fetch_list=[y.name, x.grad_name])
                x_grad = np.asarray([0.5, 0.5]).astype(np.float32)
                self.assertTrue(np.allclose(np.asarray(ret[0]), feed_x))
                self.assertTrue(np.allclose(np.asarray(ret[1]), x_grad))

    def test_forward_backward_single_tensor_output(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[2], dtype='float32')
            x.stop_gradient = False  # For test gradient
            mask = layers.data(name='mask', shape=[1], dtype='int32')

            out = program.current_block().create_var(
                dtype='float32', type=core.VarDesc.VarType.LOD_TENSOR)

            select_output(x, out, mask)
            y = select_input(out, mask)
            mean = layers.mean(y)
            append_backward(mean)

        place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        exe = Executor(place)

        feed_x = np.asarray([1.3, -1.4]).astype(np.float32)
        feed_mask = np.asarray([0]).astype(np.int32)
        ret = exe.run(program,
                      feed={'x': feed_x,
                            'mask': feed_mask},
                      fetch_list=[y.name, x.grad_name])
        x_grad = np.asarray([0.5, 0.5]).astype(np.float32)
        self.assertTrue(np.allclose(np.asarray(ret[0]), feed_x))
        self.assertTrue(np.allclose(np.asarray(ret[1]), x_grad))


if __name__ == '__main__':
    unittest.main()
