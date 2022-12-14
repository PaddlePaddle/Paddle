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
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import Program, program_guard

paddle.enable_static()


class TestTensorArrayToTensorError(unittest.TestCase):
    """Tensor_array_to_tensor error message enhance"""

    def test_errors(self):
        with program_guard(Program()):
            input_data = np.random.random((2, 4)).astype("float32")

            def test_Variable():
                fluid.layers.tensor_array_to_tensor(input=input_data)

            self.assertRaises(TypeError, test_Variable)

            def test_list_Variable():
                fluid.layers.tensor_array_to_tensor(input=[input_data])

            self.assertRaises(TypeError, test_list_Variable)


class TestLoDTensorArrayConcat(unittest.TestCase):
    """Test case for concat mode of tensor_array_to_tensor."""

    def setUp(self):
        self.op_type = "tensor_array_to_tensor"
        self.attrs = {"axis": 0}
        self.outputs = ["Out"]

    def test_get_set(self):
        scope = core.Scope()
        program = fluid.Program()
        block = program.global_block()

        input_arr = block.create_var(
            name="tmp_lod_tensor_array",
            type=core.VarDesc.VarType.LOD_TENSOR_ARRAY,
        )
        input_arr.persistable = True
        input_arr_var = scope.var('tmp_lod_tensor_array')
        input_tensor_array = input_arr_var.get_lod_tensor_array()
        self.assertEqual(0, len(input_tensor_array))

        cpu = core.CPUPlace()
        for i in range(10):
            t = core.LoDTensor()
            if i == 0:
                t.set(np.array([[i], [i]], dtype='float32'), cpu)
            else:
                t.set(np.array([[i]], dtype='float32'), cpu)
            input_tensor_array.append(t)

        self.assertEqual(10, len(input_tensor_array))

        random_grad = np.random.random_sample([11]).astype(np.float32)

        y_out = block.create_var(name="Out")
        y_out.persistable = True
        y_out_index = block.create_var(name="OutIndex")
        y_out_index.persistable = True

        y_grad_arr = block.create_var(
            name='Out@GRAD', dtype='float32', shape=[11]
        )
        y_grad_arr.persistable = True
        y_grad = scope.var('Out@GRAD')
        y_grad_tensor = y_grad.get_tensor()
        y_grad_tensor.set(random_grad, cpu)

        op = block.append_op(
            type=self.op_type,
            inputs={"X": input_arr},
            outputs={"Out": y_out, "OutIndex": y_out_index},
            attrs=self.attrs,
        )

        out_grad = block.create_var(
            name="tmp_lod_tensor_array@GRAD",
            type=core.VarDesc.VarType.LOD_TENSOR_ARRAY,
        )
        out_grad.persistable = True

        grad_op_desc_list, op_grad_to_var = core.get_grad_op_desc(
            op.desc, set(), []
        )
        grad_op_desc = grad_op_desc_list[0]
        new_op_desc = block.desc.append_op()
        new_op_desc.copy_from(grad_op_desc)
        for var_name in grad_op_desc.output_arg_names():
            block.desc.var(var_name.encode("ascii"))

        grad_op_desc.infer_var_type(block.desc)
        grad_op_desc.infer_shape(block.desc)
        for arg in grad_op_desc.output_arg_names():
            grad_var = block.desc.find_var(arg.encode("ascii"))
            grad_var.set_dtype(core.VarDesc.VarType.FP32)

        fetch_list = []
        fetch_list.append(block.var('Out'))
        fetch_list.append(block.var('OutIndex'))

        exe = fluid.Executor(fluid.CPUPlace())
        out = exe.run(program, fetch_list=fetch_list, scope=scope)
        # print ("index: ", np.array(out[1]))

        # test forward
        tensor_res = np.array(out[0])
        tensor_res_out_idx = np.array(out[1])
        tensor_gt = np.array(
            [0] + [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='float32'
        )

        self.assertEqual(len(tensor_res), len(tensor_gt))
        self.assertEqual(len(tensor_res_out_idx), 10)

        for i in range(len(tensor_res)):
            self.assertEqual(tensor_res[i], tensor_gt[i])

        for i in range(len(tensor_res_out_idx)):
            if i == 0:
                self.assertEqual(tensor_res_out_idx[i], 2)
            else:
                self.assertEqual(tensor_res_out_idx[i], 1)

        # test backward
        grad_tensor = scope.var('tmp_lod_tensor_array@GRAD')
        grad_tensor_array = grad_tensor.get_lod_tensor_array()

        self.assertEqual(10, len(grad_tensor_array))

        for i in range(len(grad_tensor_array)):
            if i == 0:
                self.assertEqual(
                    np.array(grad_tensor_array[i])[0], np.array(random_grad[i])
                )
                self.assertEqual(
                    np.array(grad_tensor_array[i])[1],
                    np.array(random_grad[i + 1]),
                )
            if i == 1:
                self.assertEqual(
                    np.array(grad_tensor_array[i]), np.array(random_grad[i + 1])
                )


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
            np.array(
                [x.shape[self.attrs["axis"]] for x in self.inputs],
                dtype="int32",
            ),
        ]
        self.input_grads = [np.ones_like(x) for x in self.inputs]
        self.set_program()
        for var in self.program.list_vars():
            # to avoid scope clearing after execution
            var.persistable = True

    def set_program(self):
        self.program = fluid.Program()
        with fluid.program_guard(self.program):
            self.array = array = paddle.tensor.create_array(dtype='float32')
            idx = fluid.layers.fill_constant(shape=[1], dtype="int64", value=0)
            for i, x in enumerate(self.inputs):
                x = fluid.layers.assign(x)
                paddle.tensor.array_write(x, idx + i, array)
            output, output_index = fluid.layers.tensor_array_to_tensor(
                input=array, **self.attrs
            )
            loss = paddle.sum(output)
            fluid.backward.append_backward(loss)
        self.output_vars = [output, output_index]

    def run_check(self, executor, scope):
        executor.run(self.program, scope=scope)
        for i, output in enumerate(self.outputs):
            np.allclose(
                np.array(scope.var(self.output_vars[i].name).get_tensor()),
                output,
                atol=0,
            )
        tensor_array_grad = scope.var(self.array.name).get_lod_tensor_array()
        for i, input_grad in enumerate(self.input_grads):
            np.allclose(np.array(tensor_array_grad[i]), input_grad, atol=0)

    def test_cpu(self):
        scope = core.Scope()
        place = core.CPUPlace()
        executor = fluid.Executor(place)
        self.run_check(executor, scope)

    def test_gpu(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            scope = core.Scope()
            executor = fluid.Executor(place)
            self.run_check(executor, scope)


class TestTensorArrayToTensorAPI(unittest.TestCase):
    def _test_case(self, inp1, inp2):
        x0 = fluid.layers.assign(inp1)
        x0.stop_gradient = False
        x1 = fluid.layers.assign(inp2)
        x1.stop_gradient = False
        i = fluid.layers.fill_constant(shape=[1], dtype="int64", value=0)
        array = paddle.tensor.create_array(dtype='float32')
        paddle.tensor.array_write(x0, i, array)
        paddle.tensor.array_write(x1, i + 1, array)
        output_stack, output_index_stack = fluid.layers.tensor_array_to_tensor(
            input=array, axis=1, use_stack=True
        )
        (
            output_concat,
            output_index_concat,
        ) = fluid.layers.tensor_array_to_tensor(
            input=array, axis=1, use_stack=False
        )
        return (
            output_stack,
            output_index_stack,
            output_concat,
            output_index_concat,
        )

    def test_case(self):
        inp0 = np.random.rand(2, 3, 4).astype("float32")
        inp1 = np.random.rand(2, 3, 4).astype("float32")

        _outs_static = self._test_case(inp0, inp1)
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        outs_static = exe.run(fetch_list=list(_outs_static))

        with fluid.dygraph.guard(place):
            outs_dynamic = self._test_case(inp0, inp1)

        for s, d in zip(outs_static, outs_dynamic):
            np.testing.assert_array_equal(s, d.numpy())

    def test_while_loop_case(self):
        with fluid.dygraph.guard():
            zero = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)
            i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=1)
            ten = fluid.layers.fill_constant(shape=[1], dtype='int64', value=10)
            array = paddle.tensor.create_array(dtype='float32')
            inp0 = np.random.rand(2, 3, 4).astype("float32")
            x0 = fluid.layers.assign(inp0)
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
            last = fluid.layers.fill_constant(shape=[1], dtype='int64', value=9)
            np.testing.assert_array_equal(
                paddle.tensor.array_read(array, last).numpy(), inp0
            )


if __name__ == '__main__':
    unittest.main()
