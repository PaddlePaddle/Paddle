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
from paddle.distributed.auto_parallel.static.pir_pass import (
    ReshardPasses,
)
from paddle.distributed.auto_parallel.static.utils import set_all_ops_op_role
from paddle.distributed.fleet.meta_optimizers.common import OpRole


class TestReshardSToR:
    def __init__(self):
        self._shape = eval(os.getenv("shape"))
        self._dtype = os.getenv("dtype")
        self._seeds = eval(os.getenv("seeds"))
        self._shard = eval(os.getenv("shard"))
        self._backend = os.getenv("backend")
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

    def run_pir_test_case(self):
        paddle.enable_static()
        if self._backend == "cpu":
            paddle.set_device("cpu")
            place = paddle.CPUPlace()
        elif self._backend == "gpu":
            place = paddle.CUDAPlace(dist.get_rank())

        BATCH_SIZE = 2
        SEQ_LEN = 4
        HIDDEN_SIZE = 8
        MP_SIZE = 2

        with paddle.pir_utils.IrGuard():
            main_program = paddle.base.Program()
            with paddle.base.program_guard(main_program):
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

                input_tensor = dist.shard_tensor(
                    w0, self._mesh, [dist.Shard(self._shard)]
                )

                reshard_tensor = paddle._C_ops.reshard(
                    input_tensor, self._mesh, [dist.Replicate()]
                )
            set_all_ops_op_role(main_program.global_block(), OpRole.Forward)
            ReshardPasses.apply_reshard_pass(main_program)
        ops = [op.name() for op in main_program.global_block().ops]
        if self._shard == 0:
            np.testing.assert_equal(main_program.num_ops(), 4)
            std_ops = [
                'builtin.parameter',
                'pd_op.data',
                'dist_op.shard_tensor',
                'pd_op.all_gather',
            ]
            np.testing.assert_equal(
                ops,
                std_ops,
            )
        elif self._shard == 1:
            np.testing.assert_equal(main_program.num_ops(), 8)
            std_ops = [
                'builtin.parameter',
                'pd_op.data',
                'dist_op.shard_tensor',
                'pd_op.all_gather',
                'pd_op.full',
                'pd_op.split_with_num',
                'pd_op.full',
                'pd_op.concat',
            ]

            np.testing.assert_equal(
                ops,
                std_ops,
            )

        for op in main_program.global_block().ops:
            if op.name() == 'pd_op.all_gather':
                # check op dist_attr
                assert op.dist_attr.num_operands() == 1
                assert op.dist_attr.num_results() == 1

                operand_dist_attr = op.dist_attr.operand(
                    0
                ).as_tensor_dist_attr()
                result_dist_attr = op.dist_attr.result(0).as_tensor_dist_attr()

                assert op.dist_attr.process_mesh == self._mesh
                assert operand_dist_attr.process_mesh == self._mesh
                if self._shard == 0:
                    assert operand_dist_attr.dims_mapping == [0, -1]
                elif self._shard == 1:
                    assert operand_dist_attr.dims_mapping == [-1, 0]
                assert operand_dist_attr.partial_status == {}

                assert result_dist_attr.process_mesh == self._mesh
                assert result_dist_attr.dims_mapping == [-1, -1]
                assert result_dist_attr.partial_status == {}

                # check op_value dist_attr
                assert op.num_results() == 1
                op_value = op.result(0)
                assert op_value.is_dense_tensor_type()
                assert op_value.is_dist_dense_tensor_type()
                assert op_value.is_dist_dense_tensor_type()
                assert op_value.dist_attr().process_mesh == self._mesh
                assert op_value.dist_attr().dims_mapping == [-1, -1]
                assert op_value.dist_attr().partial_status == {}
            elif op.name() == 'pd_op.split_with_num':
                # check op dist_attr
                assert op.dist_attr.num_operands() == 2
                assert op.dist_attr.num_results() == 1

                operand_1_dist_attr = op.dist_attr.operand(
                    0
                ).as_tensor_dist_attr()
                operand_2_dist_attr = op.dist_attr.operand(
                    1
                ).as_tensor_dist_attr()

                assert op.dist_attr.process_mesh == self._mesh
                assert operand_1_dist_attr.process_mesh == self._mesh
                assert operand_2_dist_attr.process_mesh == self._mesh

                assert operand_1_dist_attr.dims_mapping == [-1, -1]
                assert operand_2_dist_attr.dims_mapping == [-1]

                assert operand_1_dist_attr.partial_status == {}
                assert operand_2_dist_attr.partial_status == {}

                result_dist_attrs = op.dist_attr.result(0).as_array_attr()
                assert len(result_dist_attrs) == 2
                result_dist_attr_1 = result_dist_attrs[0].as_tensor_dist_attr()
                result_dist_attr_2 = result_dist_attrs[1].as_tensor_dist_attr()
                assert result_dist_attr_1.process_mesh == self._mesh
                assert result_dist_attr_1.dims_mapping == [-1, -1]
                assert result_dist_attr_1.partial_status == {}

                assert result_dist_attr_2.process_mesh == self._mesh
                assert result_dist_attr_2.dims_mapping == [-1, -1]
                assert result_dist_attr_2.partial_status == {}

                # check op_value dist_attr
                assert op.num_results() == 1
                op_value = op.result(0)
                assert op_value.is_combine()
                values = op_value.first_use().owner().results()
                for value in values:
                    assert value.dist_attr().process_mesh == self._mesh
                    assert value.dist_attr().dims_mapping == [-1, -1]
                    assert value.dist_attr().partial_status == {}
            elif op.name() == 'pd_op.concat':
                # check op dist_attr
                assert op.dist_attr.num_operands() == 2
                assert op.dist_attr.num_results() == 1

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
                assert operand_1_dist_attr_1.partial_status == {}

                assert operand_1_dist_attr_2.process_mesh == self._mesh
                assert operand_1_dist_attr_2.dims_mapping == [-1, -1]
                assert operand_1_dist_attr_2.partial_status == {}

                result_dist_attr = op.dist_attr.result(0).as_tensor_dist_attr()
                assert result_dist_attr.process_mesh == self._mesh
                assert result_dist_attr.dims_mapping == [-1, -1]
                assert result_dist_attr.partial_status == {}

                # check op_value dist_attr
                assert op.num_results() == 1
                op_value = op.result(0)
                assert op_value.is_dense_tensor_type()
                assert op_value.is_dist_dense_tensor_type()
                assert op_value.is_dist_dense_tensor_type()
                assert op_value.dist_attr().process_mesh == self._mesh
                assert op_value.dist_attr().dims_mapping == [-1, -1]
                assert op_value.dist_attr().partial_status == {}

    def run_pir_unbalanced_split_test_case(self):
        paddle.enable_static()
        if self._backend == "cpu":
            paddle.set_device("cpu")
            place = paddle.CPUPlace()
        elif self._backend == "gpu":
            place = paddle.CUDAPlace(dist.get_rank())

        BATCH_SIZE = 2
        SEQ_LEN = 4
        HIDDEN_SIZE = 9
        MP_SIZE = 2

        with paddle.pir_utils.IrGuard():
            main_program = paddle.base.Program()
            with paddle.base.program_guard(main_program):
                mesh = dist.ProcessMesh([0, 1], dim_names=['mp'])
                input = paddle.static.data(
                    name='input', shape=[BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE]
                )
                w1 = paddle.pir.core.create_parameter(
                    dtype="float32",
                    shape=[HIDDEN_SIZE, HIDDEN_SIZE],
                    name="w1",
                    initializer=paddle.nn.initializer.Uniform(),
                )

                input_tensor = dist.shard_tensor(
                    w1, self._mesh, [dist.Shard(self._shard)]
                )

                reshard_tensor = paddle._C_ops.reshard(
                    input_tensor, self._mesh, [dist.Replicate()]
                )
            ReshardPasses.apply_reshard_pass(main_program)
        # last one will pad
        need_padding = dist.get_rank() == self._mesh.process_ids[-1]
        ops = [op.name() for op in main_program.global_block().ops]
        if need_padding:
            np.testing.assert_equal(main_program.num_ops(), 18)
            std_ops = [
                'builtin.parameter',
                'pd_op.data',
                'dist_op.shard_tensor',
                'pd_op.full',
                'pd_op.full',
                'builtin.combine',
                'pd_op.concat',
                'pd_op.all_gather',
                'pd_op.full',
                'pd_op.split_with_num',
                'builtin.split',
                'pd_op.full_int_array',
                'pd_op.full',
                'pd_op.split',
                'builtin.split',
                'pd_op.full',
                'builtin.combine',
                'pd_op.concat',
            ]
            np.testing.assert_equal(
                ops,
                std_ops,
            )
        else:
            np.testing.assert_equal(main_program.num_ops(), 14)
            std_ops = [
                'builtin.parameter',
                'pd_op.data',
                'dist_op.shard_tensor',
                'pd_op.all_gather',
                'pd_op.full',
                'pd_op.split_with_num',
                'builtin.split',
                'pd_op.full_int_array',
                'pd_op.full',
                'pd_op.split',
                'builtin.split',
                'pd_op.full',
                'builtin.combine',
                'pd_op.concat',
            ]

            np.testing.assert_equal(
                ops,
                std_ops,
            )

        first_concat = True
        for op in main_program.global_block().ops:
            if op.name() == 'pd_op.all_gather':
                # check op dist_attr
                assert op.dist_attr.num_operands() == 1
                assert op.dist_attr.num_results() == 1

                operand_dist_attr = op.dist_attr.operand(
                    0
                ).as_tensor_dist_attr()
                result_dist_attr = op.dist_attr.result(0).as_tensor_dist_attr()

                assert op.dist_attr.process_mesh == self._mesh
                assert operand_dist_attr.process_mesh == self._mesh
                if self._shard == 0:
                    assert operand_dist_attr.dims_mapping == [0, -1]
                elif self._shard == 1:
                    assert operand_dist_attr.dims_mapping == [-1, 0]
                assert operand_dist_attr.partial_status == {}

                assert result_dist_attr.process_mesh == self._mesh
                assert result_dist_attr.dims_mapping == [-1, -1]
                assert result_dist_attr.partial_status == {}

                # check op_value dist_attr
                assert op.num_results() == 1
                op_value = op.result(0)
                assert op_value.is_dense_tensor_type()
                assert op_value.is_dist_dense_tensor_type()
                assert op_value.is_dist_dense_tensor_type()
                assert op_value.dist_attr().process_mesh == self._mesh
                assert op_value.dist_attr().dims_mapping == [-1, -1]
                assert op_value.dist_attr().partial_status == {}
            elif op.name() == 'pd_op.split_with_num':
                # check op dist_attr
                assert op.dist_attr.num_operands() == 2
                assert op.dist_attr.num_results() == 1

                operand_1_dist_attr = op.dist_attr.operand(
                    0
                ).as_tensor_dist_attr()
                operand_2_dist_attr = op.dist_attr.operand(
                    1
                ).as_tensor_dist_attr()

                assert op.dist_attr.process_mesh == self._mesh
                assert operand_1_dist_attr.process_mesh == self._mesh
                assert operand_2_dist_attr.process_mesh == self._mesh

                assert operand_1_dist_attr.dims_mapping == [-1, -1]
                assert operand_2_dist_attr.dims_mapping == [-1]

                assert operand_1_dist_attr.partial_status == {}
                assert operand_2_dist_attr.partial_status == {}

                result_dist_attrs = op.dist_attr.result(0).as_array_attr()
                assert len(result_dist_attrs) == 2
                result_dist_attr_1 = result_dist_attrs[0].as_tensor_dist_attr()
                result_dist_attr_2 = result_dist_attrs[1].as_tensor_dist_attr()
                assert result_dist_attr_1.process_mesh == self._mesh
                assert result_dist_attr_1.dims_mapping == [-1, -1]
                assert result_dist_attr_1.partial_status == {}

                assert result_dist_attr_2.process_mesh == self._mesh
                assert result_dist_attr_2.dims_mapping == [-1, -1]
                assert result_dist_attr_2.partial_status == {}

                # check op_value dist_attr
                assert op.num_results() == 1
                op_value = op.result(0)
                assert op_value.is_combine()
                values = op_value.first_use().owner().results()
                for value in values:
                    assert value.dist_attr().process_mesh == self._mesh
                    assert value.dist_attr().dims_mapping == [-1, -1]
                    assert value.dist_attr().partial_status == {}
            elif op.name() == 'pd_op.concat':
                if need_padding and first_concat:
                    first_concat = False
                    # check op dist_attr
                    assert op.dist_attr.num_operands() == 2
                    assert op.dist_attr.num_results() == 1

                    operand_1_dist_attrs = op.dist_attr.operand(
                        0
                    ).as_array_attr()
                    assert len(operand_1_dist_attrs) == 2

                    operand_1_dist_attr_1 = operand_1_dist_attrs[
                        0
                    ].as_tensor_dist_attr()
                    operand_1_dist_attr_2 = operand_1_dist_attrs[
                        1
                    ].as_tensor_dist_attr()
                    assert operand_1_dist_attr_1.process_mesh == self._mesh
                    if self._shard == 0:
                        assert operand_1_dist_attr_1.dims_mapping == [0, -1]
                    elif self._shard == 1:
                        assert operand_1_dist_attr_1.dims_mapping == [-1, 0]
                    assert operand_1_dist_attr_1.partial_status == {}

                    assert operand_1_dist_attr_2.process_mesh == self._mesh
                    assert operand_1_dist_attr_2.dims_mapping == [-1, -1]
                    assert operand_1_dist_attr_2.partial_status == {}

                    result_dist_attr = op.dist_attr.result(
                        0
                    ).as_tensor_dist_attr()
                    assert result_dist_attr.process_mesh == self._mesh
                    if self._shard == 0:
                        assert result_dist_attr.dims_mapping == [0, -1]
                    elif self._shard == 1:
                        assert result_dist_attr.dims_mapping == [-1, 0]
                    assert result_dist_attr.partial_status == {}

                    # check op_value dist_attr
                    assert op.num_results() == 1
                    op_value = op.result(0)
                    assert op_value.is_dense_tensor_type()
                    assert op_value.is_dist_dense_tensor_type()
                    assert op_value.is_dist_dense_tensor_type()
                    assert op_value.dist_attr().process_mesh == self._mesh
                    if self._shard == 0:
                        assert op_value.dist_attr().dims_mapping == [0, -1]
                    elif self._shard == 1:
                        assert op_value.dist_attr().dims_mapping == [-1, 0]
                    assert op_value.dist_attr().partial_status == {}
                else:
                    # check op dist_attr
                    assert op.dist_attr.num_operands() == 2
                    assert op.dist_attr.num_results() == 1

                    operand_1_dist_attrs = op.dist_attr.operand(
                        0
                    ).as_array_attr()
                    assert len(operand_1_dist_attrs) == 2

                    operand_1_dist_attr_1 = operand_1_dist_attrs[
                        0
                    ].as_tensor_dist_attr()
                    operand_1_dist_attr_2 = operand_1_dist_attrs[
                        1
                    ].as_tensor_dist_attr()
                    assert operand_1_dist_attr_1.process_mesh == self._mesh
                    assert operand_1_dist_attr_1.dims_mapping == [-1, -1]
                    assert operand_1_dist_attr_1.partial_status == {}

                    assert operand_1_dist_attr_2.process_mesh == self._mesh
                    assert operand_1_dist_attr_2.dims_mapping == [-1, -1]
                    assert operand_1_dist_attr_2.partial_status == {}

                    result_dist_attr = op.dist_attr.result(
                        0
                    ).as_tensor_dist_attr()
                    assert result_dist_attr.process_mesh == self._mesh
                    assert result_dist_attr.dims_mapping == [-1, -1]
                    assert result_dist_attr.partial_status == {}

                    # check op_value dist_attr
                    assert op.num_results() == 1
                    op_value = op.result(0)
                    assert op_value.is_dense_tensor_type()
                    assert op_value.is_dist_dense_tensor_type()
                    assert op_value.is_dist_dense_tensor_type()
                    assert op_value.dist_attr().process_mesh == self._mesh
                    assert op_value.dist_attr().dims_mapping == [-1, -1]
                    assert op_value.dist_attr().partial_status == {}


if __name__ == '__main__':
    TestReshardSToR().run_pir_test_case()
    TestReshardSToR().run_pir_unbalanced_split_test_case()
