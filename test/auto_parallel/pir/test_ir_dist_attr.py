# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.distributed.auto_parallel.api import dtensor_from_local

paddle.enable_static()

BATCH_SIZE = 2
SEQ_LEN = 4
HIDDEN_SIZE = 8
MP_SIZE = 2


class TestBuildFakeProgram(unittest.TestCase):
    def test_build_api(self):
        with paddle.pir_utils.IrGuard():
            main_program = paddle.base.Program()
            start_program = paddle.base.Program()
            with paddle.base.program_guard(main_program, start_program):
                mesh = dist.ProcessMesh([0, 1], dim_names=['mp'])
                input = paddle.static.data(
                    name='input', shape=[BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE]
                )
                w0 = paddle.pir.core.create_parameter(
                    dtype="float32",
                    shape=[HIDDEN_SIZE, HIDDEN_SIZE],
                    name="w0",
                    initializer=paddle.nn.initializer.Uniform(),
                )

                # dense tensor could not access dist tensor attribute
                with self.assertRaises(ValueError):
                    tmp = input._local_shape

                self.assertIsNone(input.dist_attr())
                self.assertIsNone(w0.dist_attr())

                dist_input = dtensor_from_local(input, mesh, [dist.Replicate()])
                dist_w0 = dtensor_from_local(w0, mesh, [dist.Replicate()])

    def test_build_replicated_program(self):
        with paddle.pir_utils.IrGuard():
            main_program = paddle.base.Program()
            start_program = paddle.base.Program()
            with paddle.base.program_guard(main_program, start_program):
                mesh = dist.ProcessMesh([0, 1], dim_names=['mp'])
                input = paddle.static.data(
                    name='input', shape=[BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE]
                )
                w0 = paddle.pir.core.create_parameter(
                    dtype="float32",
                    shape=[HIDDEN_SIZE, HIDDEN_SIZE],
                    name="w0",
                    initializer=paddle.nn.initializer.Uniform(),
                )
                self.assertTrue(input.is_dense_tensor_type())
                self.assertTrue(w0.is_dense_tensor_type())

                dist_input = dtensor_from_local(input, mesh, [dist.Replicate()])
                dist_w0 = dtensor_from_local(w0, mesh, [dist.Replicate()])
                dist_out = paddle.matmul(dist_input, dist_w0)
        self.assertTrue(dist_input.is_dist_dense_tensor_type())
        self.assertTrue(dist_w0.is_dist_dense_tensor_type())

        # check detail
        self.assertTrue(dist_input.shape == [BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE])
        self.assertTrue(w0.shape == [HIDDEN_SIZE, HIDDEN_SIZE])
        self.assertTrue(dist_input.shape == dist_input._local_shape)
        self.assertTrue(w0.shape == dist_w0._local_shape)
        self.assertTrue(dist_input.dist_attr().dims_mapping == [-1, -1, -1])
        self.assertTrue(
            isinstance(
                dist_input.dist_attr().process_mesh,
                paddle.base.libpaddle.ProcessMesh,
            )
        )
        self.assertTrue(dist_input.dist_attr().process_mesh.shape == [2])
        self.assertTrue(
            dist_input.dist_attr().process_mesh.process_ids == [0, 1]
        )
        self.assertTrue(len(dist_input.dist_attr().partial_dims) == 0)
        self.assertTrue(dist_w0.dist_attr().dims_mapping == [-1, -1])
        self.assertTrue(
            isinstance(
                dist_w0.dist_attr().process_mesh,
                paddle.base.libpaddle.ProcessMesh,
            )
        )
        self.assertTrue(dist_w0.dist_attr().process_mesh.shape == [2])
        self.assertTrue(dist_w0.dist_attr().process_mesh.process_ids == [0, 1])
        self.assertTrue(len(dist_w0.dist_attr().partial_dims) == 0)

        # matmul out
        self.assertTrue(dist_out.shape == [BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE])
        self.assertTrue(
            dist_out._local_shape == [BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE]
        )
        self.assertTrue(dist_out.dist_attr().dims_mapping == [-1, -1, -1])
        self.assertTrue(
            isinstance(
                dist_out.dist_attr().process_mesh,
                paddle.base.libpaddle.ProcessMesh,
            )
        )
        self.assertTrue(dist_out.dist_attr().process_mesh.shape == [2])
        self.assertTrue(dist_out.dist_attr().process_mesh.process_ids == [0, 1])
        self.assertTrue(len(dist_out.dist_attr().partial_dims) == 0)

    def test_build_col_parallel_program(self):
        with paddle.pir_utils.IrGuard():
            main_program = paddle.base.Program()
            start_program = paddle.base.Program()
            with paddle.base.program_guard(main_program, start_program):
                mesh = dist.ProcessMesh([0, 1], dim_names=['mp'])
                input = paddle.static.data(
                    name='input', shape=[BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE]
                )
                w0 = paddle.pir.core.create_parameter(
                    dtype="float32",
                    shape=[HIDDEN_SIZE, HIDDEN_SIZE // MP_SIZE],
                    name="w0",
                    initializer=paddle.nn.initializer.Uniform(),
                )
                self.assertTrue(input.is_dense_tensor_type())
                self.assertTrue(w0.is_dense_tensor_type())

                dist_input = dtensor_from_local(input, mesh, [dist.Replicate()])
                dist_w0 = dtensor_from_local(w0, mesh, [dist.Shard(1)])
                dist_out = paddle.matmul(dist_input, dist_w0)
        self.assertTrue(dist_input.is_dist_dense_tensor_type())
        self.assertTrue(dist_w0.is_dist_dense_tensor_type())

        # check detail
        self.assertTrue(dist_input.shape == [BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE])
        self.assertTrue(dist_w0.shape == [HIDDEN_SIZE, HIDDEN_SIZE])
        self.assertTrue(dist_input.shape == dist_input._local_shape)
        self.assertTrue(
            dist_w0._local_shape == [HIDDEN_SIZE, HIDDEN_SIZE // MP_SIZE]
        )
        self.assertTrue(dist_input.dist_attr().dims_mapping == [-1, -1, -1])
        self.assertTrue(dist_w0.dist_attr().dims_mapping == [-1, 0])
        # matmul out
        self.assertTrue(dist_out.shape == [BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE])
        self.assertTrue(
            dist_out._local_shape
            == [BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE // MP_SIZE]
        )
        self.assertTrue(dist_out.dist_attr().dims_mapping == [-1, -1, 0])
        self.assertTrue(
            isinstance(
                dist_out.dist_attr().process_mesh,
                paddle.base.libpaddle.ProcessMesh,
            )
        )
        self.assertTrue(dist_out.dist_attr().process_mesh.shape == [2])
        self.assertTrue(dist_out.dist_attr().process_mesh.process_ids == [0, 1])
        self.assertTrue(len(dist_out.dist_attr().partial_dims) == 0)

    def test_build_row_parallel_program(self):
        with paddle.pir_utils.IrGuard():
            main_program = paddle.base.Program()
            start_program = paddle.base.Program()
            with paddle.base.program_guard(main_program, start_program):
                mesh = dist.ProcessMesh([0, 1], dim_names=['mp'])
                input = paddle.static.data(
                    name='input',
                    shape=[BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE // MP_SIZE],
                )
                w0 = paddle.pir.core.create_parameter(
                    dtype="float32",
                    shape=[HIDDEN_SIZE // MP_SIZE, HIDDEN_SIZE],
                    name="w0",
                    initializer=paddle.nn.initializer.Uniform(),
                )
                self.assertTrue(input.is_dense_tensor_type())
                self.assertTrue(w0.is_dense_tensor_type())

                dist_input = dtensor_from_local(input, mesh, [dist.Shard(2)])
                dist_w0 = dtensor_from_local(w0, mesh, [dist.Shard(0)])
                dist_out = paddle.matmul(dist_input, dist_w0)
        self.assertTrue(dist_input.is_dist_dense_tensor_type())
        self.assertTrue(dist_w0.is_dist_dense_tensor_type())

        # check detail
        self.assertTrue(dist_input.shape == [BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE])
        self.assertTrue(dist_w0.shape == [HIDDEN_SIZE, HIDDEN_SIZE])
        self.assertTrue(
            dist_input._local_shape
            == [BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE // MP_SIZE]
        )
        self.assertTrue(
            dist_w0._local_shape == [HIDDEN_SIZE // MP_SIZE, HIDDEN_SIZE]
        )
        self.assertTrue(dist_input.dist_attr().dims_mapping == [-1, -1, 0])
        self.assertTrue(dist_w0.dist_attr().dims_mapping == [0, -1])
        # matmul out
        self.assertTrue(dist_out.shape == [BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE])
        self.assertTrue(
            dist_out._local_shape == [BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE]
        )
        self.assertTrue(dist_out.dist_attr().dims_mapping == [-1, -1, -1])
        self.assertTrue(
            isinstance(
                dist_out.dist_attr().process_mesh,
                paddle.base.libpaddle.ProcessMesh,
            )
        )
        self.assertTrue(dist_out.dist_attr().process_mesh.shape == [2])
        self.assertTrue(dist_out.dist_attr().process_mesh.process_ids == [0, 1])
        self.assertTrue(dist_out.dist_attr().partial_dims == {0})

    def test_build_with_shard_tensor(self):
        with paddle.pir_utils.IrGuard():
            main_program = paddle.base.Program()
            start_program = paddle.base.Program()
            with paddle.base.program_guard(main_program, start_program):
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
                w1 = paddle.pir.core.create_parameter(
                    dtype="float32",
                    shape=[HIDDEN_SIZE, HIDDEN_SIZE],
                    name="w0",
                    initializer=paddle.nn.initializer.Uniform(),
                )
                self.assertTrue(input.is_dense_tensor_type())
                self.assertTrue(w0.is_dense_tensor_type())

                dist_input = dist.shard_tensor(input, mesh, [dist.Replicate()])
                dist_w0 = dist.shard_tensor(w0, mesh, [dist.Shard(0)])
                dist_w1 = dist.shard_tensor(w1, mesh, [dist.Shard(1)])
        self.assertTrue(dist_input.is_dist_dense_tensor_type())
        self.assertTrue(dist_w0.is_dist_dense_tensor_type())

        # check global shape
        self.assertTrue(dist_input.shape == [BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE])
        self.assertTrue(dist_w0.shape == [HIDDEN_SIZE, HIDDEN_SIZE])
        self.assertTrue(dist_w1.shape == [HIDDEN_SIZE, HIDDEN_SIZE])
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

    # TODO check Dtype, layout same as densetensor
    # TODO check dims_mapping & mesh as user annotated


if __name__ == "__main__":
    unittest.main()
