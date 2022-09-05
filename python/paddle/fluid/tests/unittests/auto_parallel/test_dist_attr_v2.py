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
# limitations under the License

import unittest
import paddle
import numpy as np
import paddle.nn as nn
import paddle.static as static
from paddle.fluid.core import TensorDistAttr
from paddle.fluid.core import OperatorDistAttr

from paddle.distributed.auto_parallel.process_mesh_v2 import ProcessMesh

paddle.enable_static()


class TestDistAttr(unittest.TestCase):

    def test_tensor_dist_attr_ctor(self):
        train_program = static.Program()
        start_program = static.Program()
        with static.program_guard(train_program, start_program):
            input = static.data(name="input", shape=[2, 3], dtype='float32')
        dist_attr = TensorDistAttr(input.desc)
        self.assertEqual(dist_attr.process_mesh.empty(), True)
        self.assertEqual(dist_attr.dims_mapping, [-1, -1])
        self.assertEqual(dist_attr.batch_dim, 0)
        self.assertEqual(dist_attr.dynamic_dims, [0, 0])

        dist_attr.process_mesh = ProcessMesh([[0, 1, 2], [3, 4, 5]])
        dist_attr.dims_mapping = [0, -1]
        dist_attr.batch_dim = 1
        dist_attr.dynamic_dims = [1, 1]
        self.assertEqual(dist_attr.process_mesh,
                         ProcessMesh([[0, 1, 2], [3, 4, 5]]))
        self.assertEqual(dist_attr.dims_mapping, [0, -1])
        self.assertEqual(dist_attr.batch_dim, 1)
        self.assertEqual(dist_attr.dynamic_dims, [1, 1])
        self.assertTrue(dist_attr.verify())
        self.assertTrue(str(dist_attr), str(dist_attr))

    def test_tensor_dist_attr(self):
        train_program = static.Program()
        start_program = static.Program()
        with static.program_guard(train_program, start_program):
            input = static.data(name="input", shape=[2, 3], dtype='float32')
            input1 = static.data(name="input1", shape=[2, 3], dtype='float32')
        dist_attr = input.dist_attr
        dist_attr.process_mesh = ProcessMesh([[0, 1, 2], [3, 4, 5]])
        dist_attr.dims_mapping = [0, -1]
        dist_attr.batch_dim = 1
        dist_attr.dynamic_dims = [1, 1]
        self.assertEqual(input.dist_attr.process_mesh,
                         ProcessMesh([[0, 1, 2], [3, 4, 5]]))
        self.assertEqual(input.dist_attr.dims_mapping, [0, -1])
        self.assertEqual(input.dist_attr.batch_dim, 1)
        self.assertEqual(input.dist_attr.dynamic_dims, [1, 1])
        self.assertTrue(input.dist_attr.verify())

        input1.dist_attr = dist_attr
        self.assertEqual(input1.dist_attr.process_mesh,
                         ProcessMesh([[0, 1, 2], [3, 4, 5]]))
        self.assertEqual(input1.dist_attr.dims_mapping, [0, -1])
        self.assertEqual(input1.dist_attr.batch_dim, 1)
        self.assertEqual(input1.dist_attr.dynamic_dims, [1, 1])
        self.assertTrue(input1.dist_attr.verify())

    def test_operator_dist_attr_ctor(self):
        train_program = static.Program()
        start_program = static.Program()
        with static.program_guard(train_program, start_program):
            input = static.data(name="input", shape=[2, 3], dtype='float32')
            input1 = static.data(name="input1", shape=[3, 4], dtype='float32')
            output = paddle.matmul(input, input1)
        op = train_program.current_block().ops[0]
        process_mesh = ProcessMesh([[0, 1, 2], [3, 4, 5]])
        op_dist_attr = OperatorDistAttr(op.desc)

        op_dist_attr.process_mesh = process_mesh
        # Set the distributed attribute of input
        input_dist_attr = TensorDistAttr(input.desc)
        input_dist_attr.dims_mapping = [0, -1]
        op_dist_attr.set_input_dist_attr(input.name, input_dist_attr)
        # Set the distributed attribute of input1
        input1_dist_attr = TensorDistAttr(input1.desc)
        input1_dist_attr.dims_mapping = [-1, 1]
        op_dist_attr.set_input_dist_attr(input1.name, input1_dist_attr)
        # Set the distributed attribute of output
        output_dist_attr = TensorDistAttr(output.desc)
        output_dist_attr.dims_mapping = [0, 1]
        op_dist_attr.set_output_dist_attr(output.name, output_dist_attr)
        self.assertEqual(op_dist_attr.process_mesh, process_mesh)
        self.assertEqual(
            op_dist_attr.input_dist_attr(input.name).process_mesh, process_mesh)
        self.assertEqual(
            op_dist_attr.input_dist_attr(input1.name).process_mesh,
            process_mesh)
        self.assertEqual(
            op_dist_attr.output_dist_attr(output.name).process_mesh,
            process_mesh)
        self.assertEqual(
            op_dist_attr.input_dist_attr(input.name).dims_mapping, [0, -1])
        self.assertEqual(
            op_dist_attr.input_dist_attr(input1.name).dims_mapping, [-1, 1])
        self.assertEqual(
            op_dist_attr.output_dist_attr(output.name).dims_mapping, [0, 1])
        self.assertTrue(op_dist_attr.verify())
        self.assertTrue(str(op_dist_attr), str(op_dist_attr))

        op_dist_attr = OperatorDistAttr(op.desc)
        op_dist_attr.process_mesh = process_mesh
        # Set the distributed attribute of input directly
        input_dist_attr = op_dist_attr.input_dist_attr(input.name)
        input_dist_attr.dims_mapping = [-1, 0]
        # Set the distributed attribute of input1 directly
        input1_dist_attr = op_dist_attr.input_dist_attr(input1.name)
        input1_dist_attr.dims_mapping = [0, -1]
        # Set the distributed attribute of output directly
        output_dist_attr = op_dist_attr.output_dist_attr(output.name)
        output_dist_attr.dims_mapping = [-1, -1]
        self.assertEqual(op_dist_attr.process_mesh, process_mesh)
        self.assertEqual(input_dist_attr.process_mesh, process_mesh)
        self.assertEqual(input1_dist_attr.process_mesh, process_mesh)
        self.assertEqual(output_dist_attr.process_mesh, process_mesh)
        self.assertEqual(input_dist_attr.dims_mapping, [-1, 0])
        self.assertEqual(input1_dist_attr.dims_mapping, [0, -1])
        self.assertEqual(output_dist_attr.dims_mapping, [-1, -1])
        self.assertTrue(op_dist_attr.verify())
        self.assertTrue(str(op_dist_attr), str(op_dist_attr))

    def test_operator_dist_attr(self):
        train_program = static.Program()
        start_program = static.Program()
        with static.program_guard(train_program, start_program):
            input = static.data(name="input", shape=[2, 3], dtype='float32')
            input1 = static.data(name="input1", shape=[3, 4], dtype='float32')
            output = paddle.matmul(input, input1)
        op = train_program.current_block().ops[0]
        process_mesh = ProcessMesh([[0, 1, 2], [3, 4, 5]])
        op_dist_attr = op.dist_attr

        op_dist_attr.process_mesh = process_mesh
        # Set the distributed attribute of input
        input_dist_attr = TensorDistAttr(input.desc)
        input_dist_attr.dims_mapping = [0, -1]
        op_dist_attr.set_input_dist_attr(input.name, input_dist_attr)
        # Set the distributed attribute of input1
        input1_dist_attr = TensorDistAttr(input1.desc)
        input1_dist_attr.dims_mapping = [-1, 1]
        op_dist_attr.set_input_dist_attr(input1.name, input1_dist_attr)
        # Set the distributed attribute of output
        output_dist_attr = TensorDistAttr(output.desc)
        output_dist_attr.dims_mapping = [0, 1]
        op_dist_attr.set_output_dist_attr(output.name, output_dist_attr)

        self.assertEqual(op.desc.dist_attr.process_mesh, process_mesh)
        self.assertEqual(
            op.dist_attr.input_dist_attr(input.name).process_mesh, process_mesh)
        self.assertEqual(
            op.dist_attr.input_dist_attr(input1.name).process_mesh,
            process_mesh)
        self.assertEqual(
            op.dist_attr.input_dist_attr(input.name).dims_mapping, [0, -1])
        self.assertEqual(
            op.dist_attr.input_dist_attr(input.name).dims_mapping, [0, -1])
        self.assertEqual(
            op.desc.dist_attr.input_dist_attr(input1.name).dims_mapping,
            [-1, 1])
        self.assertEqual(
            op.dist_attr.output_dist_attr(output.name).dims_mapping, [0, 1])
        self.assertTrue(op.desc.dist_attr.verify())
        self.assertTrue(str(op_dist_attr), str(op_dist_attr))

        op.dist_attr = OperatorDistAttr(op.desc)
        self.assertEqual(op.desc.dist_attr, OperatorDistAttr(op.desc))


if __name__ == "__main__":
    unittest.main()
