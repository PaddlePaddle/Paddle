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

import os

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.distributed.auto_parallel.static.pir_pass import ReshardPasses


class TestReshardPToS:
    def __init__(self):
        self._shape = eval(os.getenv("shape"))
        self._dtype = os.getenv("dtype")
        self._seeds = eval(os.getenv("seeds"))
        self._shard = eval(os.getenv("shard"))
        self._backend = os.getenv("backend")
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        self.rank = dist.get_rank()

    def run_pir_test_case(self):
        paddle.enable_static()
        if self._backend == "gpu":
            place = paddle.CUDAPlace(dist.get_rank())

        BATCH_SIZE = 2
        SEQ_LEN = 4
        HIDDEN_SIZE = 6
        MP_SIZE = 2

        with paddle.pir_utils.IrGuard():
            main_program = paddle.base.Program()
            with paddle.base.program_guard(main_program):
                w0 = paddle.pir.core.create_parameter(
                    dtype="float32",
                    shape=[SEQ_LEN, HIDDEN_SIZE],
                    name="w0",
                    initializer=paddle.nn.initializer.Uniform(),
                )
                input_tensor = dist.shard_tensor(
                    w0, self._mesh, [dist.Partial()]
                )
                reshard_tensor = paddle._C_ops.reshard(
                    input_tensor,
                    self._mesh,
                    [dist.Shard(self._shard)],
                )
            ReshardPasses.apply_reshard_pass(main_program)

        ops = [op.name() for op in main_program.global_block().ops]

        if self._shard == 0:
            np.testing.assert_equal(main_program.num_ops(), 3)
            std_ops = [
                "builtin.parameter",
                "dist_op.shard_tensor",
                "pd_op.reduce_scatter",
            ]
            np.testing.assert_equal(
                ops,
                std_ops,
            )

        if self._shard == 1:
            np.testing.assert_equal(main_program.num_ops(), 5)
            std_ops = [
                "builtin.parameter",
                "dist_op.shard_tensor",
                "pd_op.transpose",
                "pd_op.reduce_scatter",
                "pd_op.transpose",
            ]
            np.testing.assert_equal(
                ops,
                std_ops,
            )

        for op in main_program.global_block().ops:
            if op.name() == "pd_op.reduce_scatter":
                assert op.dist_attr.num_operands() == 1
                assert op.dist_attr.num_results() == 1
                assert op.dist_attr.process_mesh == self._mesh
                op_operand_dist_attr = op.dist_attr.operand(
                    0
                ).as_tensor_dist_attr()
                assert op_operand_dist_attr.process_mesh == self._mesh
                assert op_operand_dist_attr.dims_mapping == [-1, -1]
                assert op_operand_dist_attr.partial_status == {
                    0: paddle.base.core.ReduceType.kRedSum
                }
                op_result_dist_attr = op.dist_attr.result(
                    0
                ).as_tensor_dist_attr()
                assert op_result_dist_attr.process_mesh == self._mesh
                assert op_result_dist_attr.dims_mapping == [0, -1]
                assert op_result_dist_attr.partial_status == {}

                op_value = op.result(0)
                assert op_value.is_dense_tensor_type()
                assert op_value.is_dist_dense_tensor_type()
                assert op_value.dist_attr().process_mesh == self._mesh
                assert op_value.dist_attr().dims_mapping == [0, -1]
                assert op_value.dist_attr().partial_status == {}

    def run_pir_unbalanced_split_test_case(self):
        paddle.enable_static()
        if self._backend == "gpu":
            place = paddle.CUDAPlace(dist.get_rank())

        BATCH_SIZE = 2
        SEQ_LEN = 3
        HIDDEN_SIZE = 7
        MP_SIZE = 2

        with paddle.pir_utils.IrGuard():
            main_program = paddle.base.Program()
            with paddle.base.program_guard(main_program):
                w1 = paddle.pir.core.create_parameter(
                    dtype="float32",
                    shape=[SEQ_LEN, HIDDEN_SIZE],
                    name="w1",
                    initializer=paddle.nn.initializer.Uniform(),
                )
                input_tensor1 = dist.shard_tensor(
                    w1, self._mesh, [dist.Partial()]
                )
                reshard_tensor1 = paddle._C_ops.reshard(
                    input_tensor1,
                    self._mesh,
                    [dist.Shard(self._shard)],
                )
            ReshardPasses.apply_reshard_pass(main_program)

        ops = [op.name() for op in main_program.global_block().ops]

        if self._shard == 0:
            if self.rank != self._mesh.process_ids[-1]:
                np.testing.assert_equal(main_program.num_ops(), 7)
                std_ops = [
                    "builtin.parameter",
                    "dist_op.shard_tensor",
                    "pd_op.full",
                    "pd_op.full",
                    "builtin.combine",
                    "pd_op.concat",
                    "pd_op.reduce_scatter",
                ]
                np.testing.assert_equal(
                    ops,
                    std_ops,
                )
            else:
                np.testing.assert_equal(main_program.num_ops(), 11)
                std_ops = [
                    "builtin.parameter",
                    "dist_op.shard_tensor",
                    "pd_op.full",
                    "pd_op.full",
                    "builtin.combine",
                    "pd_op.concat",
                    "pd_op.reduce_scatter",
                    'pd_op.full_int_array',
                    'pd_op.full',
                    'pd_op.split',
                    'builtin.split',
                ]
                np.testing.assert_equal(
                    ops,
                    std_ops,
                )

        if self._shard == 1:
            if self.rank != self._mesh.process_ids[-1]:
                np.testing.assert_equal(main_program.num_ops(), 9)
                std_ops = [
                    "builtin.parameter",
                    "dist_op.shard_tensor",
                    "pd_op.transpose",
                    "pd_op.full",
                    "pd_op.full",
                    "builtin.combine",
                    "pd_op.concat",
                    "pd_op.reduce_scatter",
                    "pd_op.transpose",
                ]
                np.testing.assert_equal(
                    ops,
                    std_ops,
                )
            else:
                np.testing.assert_equal(main_program.num_ops(), 13)
                std_ops = [
                    "builtin.parameter",
                    "dist_op.shard_tensor",
                    "pd_op.transpose",
                    "pd_op.full",
                    "pd_op.full",
                    "builtin.combine",
                    "pd_op.concat",
                    "pd_op.reduce_scatter",
                    'pd_op.full_int_array',
                    'pd_op.full',
                    'pd_op.split',
                    'builtin.split',
                    "pd_op.transpose",
                ]
                np.testing.assert_equal(
                    ops,
                    std_ops,
                )

        for op in main_program.global_block().ops:
            if op.name() == 'pd_op.concat':
                assert op.dist_attr.num_operands() == 2
                assert op.dist_attr.num_results() == 1
                assert op.dist_attr.process_mesh == self._mesh
                operand_1_dist_attrs = op.dist_attr.operand(0).as_array_attr()
                assert len(operand_1_dist_attrs) == 2
                operand_1_dist_attr_1 = operand_1_dist_attrs[
                    0
                ].as_tensor_dist_attr()
                operand_1_dist_attr_2 = operand_1_dist_attrs[
                    1
                ].as_tensor_dist_attr()
                assert operand_1_dist_attr_1.process_mesh == self._mesh
                assert operand_1_dist_attr_1.dims_mapping == [-1, -1]
                assert operand_1_dist_attr_1.partial_status == {
                    0: paddle.base.core.ReduceType.kRedSum
                }
                assert operand_1_dist_attr_2.process_mesh == self._mesh
                assert operand_1_dist_attr_2.dims_mapping == [-1, -1]
                assert operand_1_dist_attr_2.partial_status == {
                    0: paddle.base.core.ReduceType.kRedSum
                }
                op_result_dist_attr = op.dist_attr.result(
                    0
                ).as_tensor_dist_attr()
                assert op_result_dist_attr.process_mesh == self._mesh
                assert op_result_dist_attr.dims_mapping == [-1, -1]
                assert op_result_dist_attr.partial_status == {
                    0: paddle.base.core.ReduceType.kRedSum
                }
                op_value = op.result(0)
                assert op_value.is_dense_tensor_type()
                assert op_value.is_dist_dense_tensor_type()
                assert op_value.dist_attr().process_mesh == self._mesh
            elif op.name() == "pd_op.reduce_scatter":
                assert op.dist_attr.num_operands() == 1
                assert op.dist_attr.num_results() == 1
                assert op.dist_attr.process_mesh == self._mesh
                op_operand_dist_attr = op.dist_attr.operand(
                    0
                ).as_tensor_dist_attr()
                assert op_operand_dist_attr.process_mesh == self._mesh
                assert op_operand_dist_attr.dims_mapping == [-1, -1]
                assert op_operand_dist_attr.partial_status == {
                    0: paddle.base.core.ReduceType.kRedSum
                }
                op_result_dist_attr = op.dist_attr.result(
                    0
                ).as_tensor_dist_attr()
                assert op_result_dist_attr.process_mesh == self._mesh
                assert op_result_dist_attr.dims_mapping == [0, -1]
                assert op_result_dist_attr.partial_status == {}
                op_value = op.result(0)
                assert op_value.is_dense_tensor_type()
                assert op_value.is_dist_dense_tensor_type()
                assert op_value.dist_attr().process_mesh == self._mesh
                assert op_value.dist_attr().dims_mapping == [0, -1]
                assert op_value.dist_attr().partial_status == {}


if __name__ == '__main__':
    TestReshardPToS().run_pir_test_case()
    TestReshardPToS().run_pir_unbalanced_split_test_case()
