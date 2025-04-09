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

import unittest

import numpy as np

import paddle
from paddle import base
from paddle.base import core
from paddle.base.backward import append_backward
from paddle.base.executor import Executor
from paddle.base.framework import Program, program_guard
from paddle.static.nn.control_flow import select_input, select_output

paddle.enable_static()


class TestSplitMergeSelectedVarOps(unittest.TestCase):
    def test_forward_backward_list_output(self):
        for branch_num in range(2, 10):
            program = Program()
            with program_guard(program):
                x = paddle.static.data(name='x', shape=[-1, 2], dtype='float32')
                x.stop_gradient = False  # For test gradient
                mask = paddle.static.data(
                    name='mask', shape=[-1, 1], dtype='int32'
                )

                outputs = []
                for i in range(branch_num):
                    out = program.current_block().create_var(
                        dtype='float32',
                        shape=[2],
                        type=core.VarDesc.VarType.DENSE_TENSOR,
                    )
                    outputs.append(out)

                select_output(x, outputs, mask)
                y = select_input(outputs, mask)
                mean = paddle.mean(y)
                append_backward(mean)

            place = (
                base.CUDAPlace(0)
                if core.is_compiled_with_cuda()
                else base.CPUPlace()
            )
            exe = Executor(place)

            feed_x = np.asarray([1.3, -1.4]).astype(np.float32)
            for i in range(branch_num):
                feed_mask = np.asarray([i]).astype(np.int32)
                ret = exe.run(
                    program,
                    feed={'x': feed_x, 'mask': feed_mask},
                    fetch_list=[y.name, x.grad_name],
                )
                x_grad = np.asarray([0.5, 0.5]).astype(np.float32)
                np.testing.assert_allclose(
                    np.asarray(ret[0]), feed_x, rtol=1e-05
                )
                np.testing.assert_allclose(
                    np.asarray(ret[1]), x_grad, rtol=1e-05
                )


class TestSelectInputOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            mask = paddle.static.data(name='mask', shape=[-1, 1], dtype='int32')
            in1 = paddle.static.data(name='in1', shape=[-1, 1], dtype='int32')

            # 1. The type of inputs in select_input must be list or tuple.
            def test_inputs_type():
                select_input(1, mask)

            self.assertRaises(TypeError, test_inputs_type)

            # 2. The type of mask in select_input must be Variable.
            def test_mask_type():
                select_input([in1], mask=1)

            self.assertRaises(TypeError, test_mask_type)

            # 3. The dtype of mask in select_input must be int32 or int64.
            def test_mask_dtype():
                mask = paddle.static.data(
                    name='mask2', shape=[-1, 1], dtype='float32'
                )
                select_input([in1], mask)

            self.assertRaises(TypeError, test_mask_dtype)


class TestSelectOutput_Error(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            in1 = paddle.static.data(name='in1', shape=[-1, 1], dtype='int32')
            mask_int32 = paddle.static.data(
                name='mask_int32', shape=[-1, 1], dtype='int32'
            )
            mask_float32 = paddle.static.data(
                name='mask_float32', shape=[-1, 1], dtype='float32'
            )
            out1 = paddle.static.data(name='out1', shape=[-1, 1], dtype='int32')

            # 1. The type of input in select_output must Variable.
            def test_input_type():
                select_output(1, [out1], mask_int32)

            self.assertRaises(TypeError, test_input_type)

            # 2. The type of mask in select_output must be Variable.
            def test_mask_type():
                select_output(in1, [out1], mask=1)

            self.assertRaises(TypeError, test_mask_type)

            # 3. The dtype of mask in select_output must be int32 or int64.
            def test_mask_dtype():
                select_output(in1, [out1], mask=mask_float32)

            self.assertRaises(TypeError, test_mask_dtype)

            # 4. The type of mask in select_output must be list or tuple.
            def test_outputs_type():
                select_output(in1, out1, mask=mask_int32)

            self.assertRaises(TypeError, test_outputs_type)


if __name__ == '__main__':
    unittest.main()
