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
                w0 = paddle.pir.core.create_parameter(
                    dtype="float32",
                    shape=[HIDDEN_SIZE, HIDDEN_SIZE],
                    name="w0",
                    initializer=paddle.nn.initializer.Uniform(),
                )
                w1 = paddle.create_parameter(
                    dtype="float32",
                    shape=[HIDDEN_SIZE, HIDDEN_SIZE],
                    name="w1",
                    initializer=paddle.nn.initializer.Uniform(),
                )
                self.assertTrue(input.is_dense_tensor_type())
                self.assertTrue(w0.is_dense_tensor_type())

                dist_input = dist.shard_tensor(input, mesh, [dist.Replicate()])
                dist_w0 = dist.shard_tensor(w0, mesh, [dist.Shard(0)])
                dist_w1 = dist.shard_tensor(w1, mesh, [dist.Shard(1)])

        print(f'main_program: {main_program}')
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

        dist_input_op_dist_attr = dist_input.get_defining_op().dist_attr()
        attrs_op_dist_attr = (
            dist_input.get_defing_op().attrs().get("op_dist_attr")
        )
        # #check attrs
        self.assertEqual(dist_input_op_dist_attr, attrs_op_dist_attr)

        self.assertEqual(dist_input_op_dist_attr.process_mesh(), mesh)
        self.assertEqual(dist_input_op_dist_attr.num_operand_dist_attrs(), 0)
        self.assertEqual(dist_input_op_dist_attr.num_result_dist_attrs(), 1)

        dist_w0_op_dist_attr = dist_w0.get_defining_op().dist_attr()
        self.assertEqual(dist_w0_op_dist_attr.process_mesh(), mesh)
        self.assertEqual(dist_w0_op_dist_attr.num_operand_dist_attrs(), 0)
        self.assertEqual(dist_w0_op_dist_attr.num_result_dist_attrs(), 1)

        dist_w1_op_dist_attr = dist_w1.get_defining_op().dist_attr()
        self.assertEqual(dist_w1_op_dist_attr.process_mesh(), mesh)
        self.assertEqual(dist_w1_op_dist_attr.num_operand_dist_attrs(), 0)
        self.assertEqual(dist_w1_op_dist_attr.num_result_dist_attrs(), 1)

        # check op result dist_attr
        self.assertEqual(
            dist_input_op_dist_attr.result_dist_attr(0).process_mesh(), mesh
        )
        self.assertEqual(
            dist_input_op_dist_attr.result_dist_attr(0).dims_mapping,
            [-1, -1, -1],
        )

        self.assertEqual(
            dist_w0_op_dist_attr.result_dist_attr(0).process_mesh, mesh
        )
        self.assertEqual(
            dist_w0_op_dist_attr.result_dist_attr(0).dims_mapping, [0, -1]
        )

        self.assertEqual(
            dist_w1_op_dist_attr.result_dist_attr(0).process_mesh, mesh
        )
        self.assertEqual(
            dist_w1_op_dist_attr.result_dist_attr(0).dims_mapping, [-1, 0]
        )

        # check value dist_attr
        dist_input_dist_attr = dist_input.dist_attr()
        self.assertEqual(
            dist_input_op_dist_attr.result_dist_attr(0), dist_input_dist_attr
        )
        self.assertEqual(dist_input_dist_attr.process_mesh(), mesh)
        self.assertEqual(dist_input_dist_attr.dims_mapping, [-1, -1, -1])

        dist_w0_dist_attr = dist_w0.dist_attr()
        self.assertEqual(
            dist_w0_op_dist_attr.result_dist_attr(0), dist_w0_dist_attr
        )
        self.assertEqual(dist_w0_dist_attr.process_mesh(), mesh)
        self.assertEqual(dist_w0_dist_attr.dims_mapping, [0, -1])

        dist_w1_dist_attr = dist_w1.dist_attr()
        self.assertEqual(
            dist_w1_op_dist_attr.result_dist_attr(0), dist_w1_dist_attr
        )
        self.assertEqual(dist_w1_dist_attr.process_mesh(), mesh)
        self.assertEqual(dist_w1_dist_attr.dims_mapping, [-1, 0])


if __name__ == "__main__":
    unittest.main()
