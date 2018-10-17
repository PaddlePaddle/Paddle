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
import numpy
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.op import Operator
from paddle.fluid.executor import Executor


class TestLoDTensorArrayConcat(unittest.TestCase):
    def setUp(self):
        self.op_type = "lod_tensor_array_concat"
        self.attrs = {"axis": 0}
        self.outputs = ["Out"]

    def test_get_set(self):
        scope = core.Scope()
        program = fluid.Program()
        block = program.global_block()

        input_arr = block.create_var(
            name="tmp_lod_tensor_array",
            type=core.VarDesc.VarType.LOD_TENSOR_ARRAY)
        input_arr.persistable = True
        input_arr_var = scope.var('tmp_lod_tensor_array')
        input_tensor_array = input_arr_var.get_lod_tensor_array()
        self.assertEqual(0, len(input_tensor_array))

        cpu = core.CPUPlace()
        for i in range(10):
            t = core.LoDTensor()
            t.set(numpy.array([i], dtype='float32'), cpu)
            t.set_recursive_sequence_lengths([[1]])
            input_tensor_array.append(t)

        self.assertEqual(10, len(input_tensor_array))

        random_grad = numpy.random.random_sample([10]).astype(numpy.float32)

        y_out = block.create_var(name="Out")
        y_out.persistable = True

        y_grad_arr = block.create_var(
            name='Out@GRAD', dtype='float32', shape=[10])
        y_grad_arr.persistable = True
        y_grad = scope.var('Out@GRAD')
        y_grad_tensor = y_grad.get_tensor()
        y_grad_tensor.set(random_grad, cpu)

        op = block.append_op(
            type=self.op_type,
            inputs={"X": input_arr},
            outputs={"Out": y_out},
            attrs=self.attrs)

        out_grad = block.create_var(
            name="tmp_lod_tensor_array@GRAD",
            type=core.VarDesc.VarType.LOD_TENSOR_ARRAY)
        out_grad.persistable = True

        grad_op_desc_list, op_grad_to_var = core.get_grad_op_desc(op.desc,
                                                                  set(), [])
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

        exe = fluid.Executor(fluid.CPUPlace())
        out = exe.run(program, fetch_list=fetch_list, scope=scope)

        # test forward
        tensor = out[0]
        tensor_res = numpy.array(tensor)
        tensor_gt = numpy.array(range(10), dtype='float32')

        self.assertEqual(len(tensor_res), len(tensor_gt))

        for i in range(len(tensor_res)):
            self.assertEqual(tensor_res[i], tensor_gt[i])

        # test backward
        grad_tensor = scope.var('tmp_lod_tensor_array@GRAD')
        grad_tensor_array = grad_tensor.get_lod_tensor_array()

        self.assertEqual(len(random_grad), len(grad_tensor_array))
        for i in range(len(grad_tensor_array)):
            self.assertEqual(
                numpy.array(grad_tensor_array[i]), numpy.array(random_grad[i]))


if __name__ == '__main__':
    unittest.main()
