# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.distributed as dist
from paddle.base.libpaddle.pir import apply_dist2dense_pass
from paddle.distributed.auto_parallel.static.mix_to_dist_pass import (
    apply_mix2dist_pass,
)

BATCH_SIZE = 2
SEQ_LEN = 4
HIDDEN_SIZE = 8
MP_SIZE = 2


class TestBuildFakeProgram(unittest.TestCase):
    def test_build_with_shard_tensor(self):
        paddle.enable_static()
        with paddle.pir_utils.IrGuard():
            main_program = paddle.base.Program()
            with paddle.base.program_guard(main_program):
                mesh = dist.ProcessMesh([0, 1], dim_names=['mp'])
                input = paddle.static.data(
                    name='input',
                    shape=[BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE],
                )
                w0 = paddle.create_parameter(
                    dtype="float32",
                    shape=[HIDDEN_SIZE, HIDDEN_SIZE],
                    name="w0",
                    default_initializer=paddle.nn.initializer.Uniform(),
                )
                w1 = paddle.create_parameter(
                    dtype="float32",
                    shape=[HIDDEN_SIZE, HIDDEN_SIZE],
                    name="w1",
                    default_initializer=paddle.nn.initializer.Uniform(),
                )
                self.assertTrue(input.is_dense_tensor_type())
                self.assertTrue(w0.is_dense_tensor_type())

                dist_input = dist.shard_tensor(input, mesh, [dist.Replicate()])
                dist_w0 = dist.shard_tensor(w0, mesh, [dist.Shard(0)])
                dist_w1 = dist.shard_tensor(w1, mesh, [dist.Shard(1)])

        self.assertTrue(main_program.num_ops() == 6)

        self.assertFalse(input.use_empty())
        self.assertFalse(w0.use_empty())
        self.assertFalse(w1.use_empty())

        self.assertTrue(dist_input.use_empty())
        self.assertTrue(dist_w0.use_empty())
        self.assertTrue(dist_w1.use_empty())

        self.assertTrue(w0.is_dense_tensor_type())
        self.assertTrue(w1.is_dense_tensor_type())
        self.assertTrue(input.is_dense_tensor_type())

        # check dist type
        self.assertTrue(dist_input.is_dist_dense_tensor_type())
        self.assertTrue(dist_w0.is_dist_dense_tensor_type())
        self.assertTrue(dist_w1.is_dist_dense_tensor_type())

        # check global shape
        self.assertEqual(dist_input.shape, [BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE])
        self.assertEqual(dist_w0.shape, [HIDDEN_SIZE, HIDDEN_SIZE])
        self.assertEqual(dist_w1.shape, [HIDDEN_SIZE, HIDDEN_SIZE])
        # check local shape
        self.assertTrue(
            dist_input._local_shape == dist_input.shape
        )  # replicated, local = global
        self.assertTrue(
            dist_w0._local_shape == [HIDDEN_SIZE // MP_SIZE, HIDDEN_SIZE]
        )  # sharded, local != global, sharded by mesh size
        self.assertTrue(
            dist_w1._local_shape == [HIDDEN_SIZE, HIDDEN_SIZE // MP_SIZE]
        )  # sharded, local != global, sharded by mesh size

        # check op dist_attr
        self.assertFalse(input.get_defining_op().has_attr("op_dist_attr"))
        self.assertFalse(w0.get_defining_op().has_attr("op_dist_attr"))
        self.assertFalse(w1.get_defining_op().has_attr("op_dist_attr"))

        dist_input_op_dist_attr = dist_input.get_defining_op().dist_attr
        # #check attrs

        self.assertEqual(dist_input_op_dist_attr.process_mesh, mesh)
        self.assertEqual(dist_input_op_dist_attr.num_operands(), 0)
        self.assertEqual(dist_input_op_dist_attr.num_results(), 1)

        dist_w0_op_dist_attr = dist_w0.get_defining_op().dist_attr
        self.assertEqual(dist_w0_op_dist_attr.process_mesh, mesh)
        self.assertEqual(dist_w0_op_dist_attr.num_operands(), 0)
        self.assertEqual(dist_w0_op_dist_attr.num_results(), 1)

        dist_w1_op_dist_attr = dist_w1.get_defining_op().dist_attr
        self.assertEqual(dist_w1_op_dist_attr.process_mesh, mesh)
        self.assertEqual(dist_w1_op_dist_attr.num_operands(), 0)
        self.assertEqual(dist_w1_op_dist_attr.num_results(), 1)

        attrs_op_dist_attr = (
            dist_input.get_defining_op().attrs().get("op_dist_attr")
        )
        self.assertEqual(attrs_op_dist_attr.process_mesh, mesh)

        # check op result dist_attr
        tensor_dist_attr = dist_input_op_dist_attr.result(
            0
        ).as_tensor_dist_attr()
        self.assertEqual(tensor_dist_attr.process_mesh, mesh)
        self.assertEqual(
            tensor_dist_attr.dims_mapping,
            [-1, -1, -1],
        )
        tensor_dist_attr = dist_w0_op_dist_attr.result(0).as_tensor_dist_attr()
        self.assertEqual(tensor_dist_attr.process_mesh, mesh)
        self.assertEqual(tensor_dist_attr.dims_mapping, [0, -1])

        tensor_dist_attr = dist_w1_op_dist_attr.result(0).as_tensor_dist_attr()
        self.assertEqual(tensor_dist_attr.process_mesh, mesh)
        self.assertEqual(tensor_dist_attr.dims_mapping, [-1, 0])

        # check value dist_attr
        self.assertEqual(dist_input.dist_attr().process_mesh, mesh)
        self.assertEqual(dist_input.dist_attr().dims_mapping, [-1, -1, -1])

        self.assertEqual(dist_w0.dist_attr().process_mesh, mesh)
        self.assertEqual(dist_w0.dist_attr().dims_mapping, [0, -1])

        self.assertEqual(dist_w1.dist_attr().process_mesh, mesh)
        self.assertEqual(dist_w1.dist_attr().dims_mapping, [-1, 0])

    def test_build_with_apply_mix2dist_pass(self):
        paddle.enable_static()
        with paddle.pir_utils.IrGuard():
            main_program = paddle.base.Program()
            with paddle.base.program_guard(main_program):
                mesh = dist.ProcessMesh([0, 1], dim_names=['dp'])
                input1 = paddle.randint(low=0, high=1000, shape=[8, 4])
                output1 = dist.shard_tensor(input1, mesh, [dist.Shard(0)])

                input2 = paddle.randn([4, 8])
                output2 = dist.shard_tensor(input2, mesh, [dist.Shard(1)])

                self.assertTrue(input1.is_dense_tensor_type())
                self.assertTrue(input2.is_dense_tensor_type())

        self.assertTrue(main_program.num_ops() == 6)

        self.assertFalse(input1.use_empty())
        self.assertFalse(input2.use_empty())

        self.assertTrue(output1.use_empty())
        self.assertTrue(output2.use_empty())

        self.assertFalse(input1.get_defining_op().has_attr("op_dist_attr"))
        self.assertFalse(input2.get_defining_op().has_attr("op_dist_attr"))

        # check dist type
        self.assertTrue(output1.is_dist_dense_tensor_type())
        self.assertTrue(output2.is_dist_dense_tensor_type())

        # run apply_mix2dist_pass
        apply_mix2dist_pass(main_program)

        # after apply_mix2dist_pass, the program changed
        self.assertTrue(main_program.num_ops() == 4)

        self.assertTrue(input1.is_dist_dense_tensor_type())
        self.assertTrue(input2.is_dist_dense_tensor_type())

        self.assertTrue(input1.get_defining_op().has_attr("op_dist_attr"))
        self.assertTrue(input2.get_defining_op().has_attr("op_dist_attr"))

        # check op result dist_attr
        input1_op_dist_attr = input1.get_defining_op().dist_attr
        tensor_dist_attr = input1_op_dist_attr.result(0).as_tensor_dist_attr()
        self.assertEqual(tensor_dist_attr.process_mesh, mesh)
        self.assertEqual(tensor_dist_attr.dims_mapping, [0, -1])

        input2_op_dist_attr = input2.get_defining_op().dist_attr
        tensor_dist_attr = input2_op_dist_attr.result(0).as_tensor_dist_attr()
        self.assertEqual(tensor_dist_attr.process_mesh, mesh)
        self.assertEqual(tensor_dist_attr.dims_mapping, [-1, 0])

        # check value dist_attr
        self.assertEqual(input1.dist_attr().process_mesh, mesh)
        self.assertEqual(input1.dist_attr().dims_mapping, [0, -1])

        self.assertEqual(input2.dist_attr().process_mesh, mesh)
        self.assertEqual(input2.dist_attr().dims_mapping, [-1, 0])

        # check full_int_array op result dist_attr
        input1_shape = input1.get_defining_op().operand_source(0)
        input1_shape_op_dist_attr = input1_shape.get_defining_op().dist_attr
        tensor_dist_attr = input1_shape_op_dist_attr.result(
            0
        ).as_tensor_dist_attr()
        self.assertEqual(tensor_dist_attr.process_mesh, mesh)
        self.assertEqual(tensor_dist_attr.dims_mapping, [-1])

        input2_shape = input2.get_defining_op().operand_source(0)
        input2_shape_op_dist_attr = input2_shape.get_defining_op().dist_attr
        tensor_dist_attr = input2_shape_op_dist_attr.result(
            0
        ).as_tensor_dist_attr()
        self.assertEqual(tensor_dist_attr.process_mesh, mesh)
        self.assertEqual(tensor_dist_attr.dims_mapping, [-1])

        # check shape value dist_attr
        self.assertEqual(input1_shape.dist_attr().process_mesh, mesh)
        self.assertEqual(input1_shape.dist_attr().dims_mapping, [-1])

        self.assertEqual(input2_shape.dist_attr().process_mesh, mesh)
        self.assertEqual(input2_shape.dist_attr().dims_mapping, [-1])

    def test_build_with_apply_dist2dense_pass(self):
        paddle.enable_static()
        with paddle.pir_utils.IrGuard():
            main_program = paddle.base.Program()
            with paddle.base.program_guard(main_program):
                mesh = dist.ProcessMesh([0, 1], dim_names=['dp'])
                input1 = paddle.randint(low=0, high=1000, shape=[8, 4])
                output1 = dist.shard_tensor(input1, mesh, [dist.Shard(0)])

                input2 = paddle.randn([4, 8])
                output2 = dist.shard_tensor(input2, mesh, [dist.Shard(1)])

                self.assertTrue(input1.is_dense_tensor_type())
                self.assertTrue(input2.is_dense_tensor_type())

        self.assertTrue(main_program.num_ops() == 6)

        self.assertFalse(input1.use_empty())
        self.assertFalse(input2.use_empty())

        self.assertTrue(output1.use_empty())
        self.assertTrue(output2.use_empty())

        self.assertFalse(input1.get_defining_op().has_attr("op_dist_attr"))
        self.assertFalse(input2.get_defining_op().has_attr("op_dist_attr"))

        # check dist type
        self.assertTrue(output1.is_dist_dense_tensor_type())
        self.assertTrue(output2.is_dist_dense_tensor_type())

        # run apply_mix2dist_pass and apply_dist2dense_pass
        apply_mix2dist_pass(main_program)
        apply_dist2dense_pass(main_program)

        # after apply_mix2dist_pass, the program changed
        # and after apply_dist2dense_pass, the operator in program do not have dist_attr
        self.assertTrue(main_program.num_ops() == 4)

        self.assertTrue(input1.is_dense_tensor_type())
        self.assertTrue(input2.is_dense_tensor_type())

        self.assertFalse(input1.get_defining_op().has_attr("op_dist_attr"))
        self.assertFalse(input2.get_defining_op().has_attr("op_dist_attr"))

        # check shape attribute of full_int_array op
        input1_shape = input1.get_defining_op().operand_source(0)
        input1_shape_op = input1_shape.get_defining_op()
        self.assertFalse(input1_shape_op.has_attr("op_dist_attr"))
        input1_shape_op_attr = input1_shape_op.attrs()
        self.assertEqual(input1_shape_op_attr['value'], [4, 4])

        input2_shape = input2.get_defining_op().operand_source(0)
        input2_shape_op = input2_shape.get_defining_op()
        self.assertFalse(input2_shape_op.has_attr("op_dist_attr"))
        input2_shape_op_attr = input2_shape_op.attrs()
        self.assertEqual(input2_shape_op_attr['value'], [4, 4])

        # check shape of input1 and input2
        self.assertEqual(input1.shape, [4, 4])
        self.assertEqual(input2.shape, [4, 4])


if __name__ == "__main__":
    unittest.main()
