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
from paddle.base import Program, core, program_guard
from paddle.pir_utils import test_with_pir_api
from paddle.tensor.manipulation import tensor_array_to_tensor

paddle.enable_static()


class TestTensorArrayToTensorError(unittest.TestCase):
    """Tensor_array_to_tensor error message enhance"""

    def test_errors(self):
        with program_guard(Program()):
            input_data = np.random.random((2, 4)).astype("float32")

            def test_Variable():
                tensor_array_to_tensor(input=input_data)

            self.assertRaises(TypeError, test_Variable)

            def test_list_Variable():
                tensor_array_to_tensor(input=[input_data])

            self.assertRaises(TypeError, test_list_Variable)


class TestLoDTensorArrayStack(unittest.TestCase):
    """Test case for stack mode of tensor_array_to_tensor."""

    def setUp(self):
        self.op_type = "tensor_array_to_tensor"
        self.attrs = {"axis": 1, "use_stack": True}
        self.inputs = [
            np.random.rand(2, 3, 4).astype("float32"),
            np.random.rand(2, 3, 4).astype("float32"),
            np.random.rand(2, 3, 4).astype("float32"),
        ]
        self.outputs = [
            np.stack(self.inputs, axis=self.attrs["axis"]),
        ]
        self.input_grads = [np.ones_like(x) for x in self.inputs]
        self.set_program()
        for var in self.program.list_vars():
            # to avoid scope clearing after execution
            var.persistable = True

    def set_program(self):
        self.program = base.Program()
        with base.program_guard(self.program):
            self.array = array = paddle.tensor.create_array(dtype='float32')
            idx = paddle.tensor.fill_constant(shape=[1], dtype="int64", value=0)
            for i, x in enumerate(self.inputs):
                x = paddle.assign(x)
                paddle.tensor.array_write(x, idx + i, array)
            output, output_index = tensor_array_to_tensor(
                input=array, **self.attrs
            )
            loss = paddle.sum(output)
            base.backward.append_backward(loss)
        self.output_vars = [output]

    def run_check(self, executor, scope):
        result = executor.run(
            self.program, fetch_list=self.output_vars, scope=scope
        )
        for i, output in enumerate(self.outputs):
            np.allclose(result[i], output, atol=0)
        if not paddle.framework.use_pir_api():
            tensor_array_grad = scope.var(
                self.array.name
            ).get_lod_tensor_array()
            for i, input_grad in enumerate(self.input_grads):
                np.allclose(np.array(tensor_array_grad[i]), input_grad, atol=0)

    def test_cpu(self):
        scope = core.Scope()
        place = core.CPUPlace()
        executor = base.Executor(place)
        self.run_check(executor, scope)

    def test_gpu(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            scope = core.Scope()
            executor = base.Executor(place)
            self.run_check(executor, scope)


class TestTensorArrayToTensorAPI(unittest.TestCase):
    def _test_case(self, inp1, inp2):
        x0 = paddle.assign(inp1)
        x0.stop_gradient = False
        x1 = paddle.assign(inp2)
        x1.stop_gradient = False
        i = paddle.tensor.fill_constant(shape=[1], dtype="int64", value=0)
        array = paddle.tensor.create_array(dtype='float32')
        paddle.tensor.array_write(x0, i, array)
        paddle.tensor.array_write(x1, i + 1, array)
        output_stack, output_index_stack = tensor_array_to_tensor(
            input=array, axis=1, use_stack=True
        )
        (
            output_concat,
            output_index_concat,
        ) = tensor_array_to_tensor(input=array, axis=1, use_stack=False)
        return (
            output_stack,
            output_concat,
        )

    def test_case(self):
        inp0 = np.random.rand(2, 3, 4).astype("float32")
        inp1 = np.random.rand(2, 3, 4).astype("float32")

        _outs_static = self._test_case(inp0, inp1)
        place = base.CPUPlace()
        exe = base.Executor(place)
        outs_static = exe.run(fetch_list=list(_outs_static))

        with base.dygraph.guard(place):
            outs_dynamic = self._test_case(inp0, inp1)

        for s, d in zip(outs_static, outs_dynamic):
            np.testing.assert_array_equal(s, d.numpy())

    @test_with_pir_api
    def test_while_loop_case(self):
        with base.dygraph.guard():
            zero = paddle.tensor.fill_constant(
                shape=[1], dtype='int64', value=0
            )
            i = paddle.tensor.fill_constant(shape=[1], dtype='int64', value=1)
            ten = paddle.tensor.fill_constant(
                shape=[1], dtype='int64', value=10
            )
            array = paddle.tensor.create_array(dtype='float32')
            inp0 = np.random.rand(2, 3, 4).astype("float32")
            x0 = paddle.assign(inp0)
            paddle.tensor.array_write(x0, zero, array)

            def cond(i, end, array):
                return paddle.less_than(i, end)

            def body(i, end, array):
                prev = paddle.tensor.array_read(array, i - 1)
                paddle.tensor.array_write(prev, i, array)
                return i + 1, end, array

            _, _, array = paddle.static.nn.while_loop(
                cond, body, [i, ten, array]
            )

            self.assertTrue(paddle.tensor.array_length(array), 10)
            last = paddle.tensor.fill_constant(
                shape=[1], dtype='int64', value=9
            )
            np.testing.assert_array_equal(
                paddle.tensor.array_read(array, last).numpy(), inp0
            )


class TestPirArrayOp(unittest.TestCase):
    def test_array(self):
        paddle.enable_static()
        with paddle.pir_utils.IrGuard():
            main_program = paddle.static.Program()
            with paddle.static.program_guard(main_program):
                x = paddle.full(shape=[1, 3], fill_value=5, dtype="float32")
                y = paddle.full(shape=[1, 3], fill_value=6, dtype="float32")
                array = paddle.tensor.create_array(
                    dtype="float32", initialized_list=[x, y]
                )
                (
                    output,
                    output_index,
                ) = paddle.tensor.manipulation.tensor_array_to_tensor(
                    input=array, axis=1, use_stack=False
                )

            place = (
                paddle.base.CPUPlace()
                if not paddle.base.core.is_compiled_with_cuda()
                else paddle.base.CUDAPlace(0)
            )
            exe = paddle.base.Executor(place)
            [fetched_out0, fetched_out1] = exe.run(
                main_program, feed={}, fetch_list=[output, output_index]
            )

        np.testing.assert_array_equal(
            fetched_out0,
            np.array([[5.0, 5.0, 5.0, 6.0, 6.0, 6.0]], dtype="float32"),
        )
        np.testing.assert_array_equal(
            fetched_out1, np.array([3, 3], dtype="int32")
        )

    @test_with_pir_api
    def test_array_concat_backward(self):
        paddle.enable_static()
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            x = paddle.full(shape=[1, 4], fill_value=5, dtype="float32")
            y = paddle.full(shape=[1, 4], fill_value=6, dtype="float32")
            x.stop_gradient = False
            y.stop_gradient = False

            array = paddle.tensor.create_array(
                dtype="float32", initialized_list=[x, y]
            )
            array.stop_gradient = False
            (
                output,
                output_index,
            ) = paddle.tensor.manipulation.tensor_array_to_tensor(
                input=array, axis=1, use_stack=False
            )

            loss = paddle.mean(output)
            dout = paddle.base.gradients(loss, [x, y])

        place = (
            paddle.base.CPUPlace()
            if not paddle.base.core.is_compiled_with_cuda()
            else paddle.base.CUDAPlace(0)
        )
        exe = paddle.base.Executor(place)
        [fetched_out0, fetched_out1, fetched_out2] = exe.run(
            main_program, feed={}, fetch_list=[output, dout[0], dout[1]]
        )

        np.testing.assert_array_equal(
            fetched_out0,
            np.array(
                [[5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0]], dtype="float32"
            ),
        )
        np.testing.assert_array_equal(
            fetched_out1,
            np.array([[0.125, 0.125, 0.125, 0.125]], dtype="float32"),
        )
        np.testing.assert_array_equal(
            fetched_out2,
            np.array([[0.125, 0.125, 0.125, 0.125]], dtype="float32"),
        )

    @test_with_pir_api
    def test_array_stack_backward(self):
        paddle.enable_static()
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            x = paddle.full(shape=[1, 4], fill_value=5, dtype="float32")
            y = paddle.full(shape=[1, 4], fill_value=6, dtype="float32")
            x.stop_gradient = False
            y.stop_gradient = False

            array = paddle.tensor.create_array(
                dtype="float32", initialized_list=[x, y]
            )
            array.stop_gradient = False
            (
                output,
                output_index,
            ) = paddle.tensor.manipulation.tensor_array_to_tensor(
                input=array, axis=0, use_stack=True
            )

            loss = paddle.mean(output)
            dout = paddle.base.gradients(loss, [x, y])

        place = (
            paddle.base.CPUPlace()
            if not paddle.base.core.is_compiled_with_cuda()
            else paddle.base.CUDAPlace(0)
        )
        exe = paddle.base.Executor(place)
        [fetched_out0, fetched_out1, fetched_out2] = exe.run(
            main_program, feed={}, fetch_list=[output, dout[0], dout[1]]
        )

        np.testing.assert_array_equal(
            fetched_out0,
            np.array(
                [[[5.0, 5.0, 5.0, 5.0]], [[6.0, 6.0, 6.0, 6.0]]],
                dtype="float32",
            ),
        )
        np.testing.assert_array_equal(
            fetched_out1,
            np.array([[0.125, 0.125, 0.125, 0.125]], dtype="float32"),
        )
        np.testing.assert_array_equal(
            fetched_out2,
            np.array([[0.125, 0.125, 0.125, 0.125]], dtype="float32"),
        )


if __name__ == '__main__':
    unittest.main()
