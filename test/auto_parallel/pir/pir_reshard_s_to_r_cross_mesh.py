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

import os

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.base import core

from paddle.distributed.auto_parallel.static.pir_pass import (
    apply_reshard_pass,
)

class TestReshardSToRCrossMesh:
    def __init__(self):
        self._shape = eval(os.getenv("shape"))
        self._dtype = os.getenv("dtype")
        self._seeds = eval(os.getenv("seeds"))
        self._shard = eval(os.getenv("shard"))
        self._backend = os.getenv("backend")

        self._in_mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        self._out_mesh = dist.ProcessMesh([1, 0], dim_names=["x"])

    def run_pir_test_case(self):
        paddle.enable_static()
        BATCH_SIZE = 2
        SEQ_LEN = 4
        HIDDEN_SIZE = 8
        MP_SIZE = 2

        with paddle.pir_utils.IrGuard():
            main_program = paddle.base.Program()
            with paddle.base.program_guard(main_program):
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
                    w0, self._in_mesh, [dist.Shard(self._shard)]
                )
                reshard_tensor = paddle._pir_ops.reshard(
                    input_tensor, self._out_mesh, [dist.Replicate()]
                )
            print(f'debug main_program: {main_program}')
            dist_program = apply_reshard_pass(main_program)

        print(f'debug dist_program: {dist_program}')
        ops = [op.name() for op in dist_program.global_block().ops]
        if paddle.distributed.get_rank() == 0:
            np.testing.assert_equal(dist_program.num_ops(), 4)
            std_ops = [
                'builtin.parameter',
                'pd_op.data',
                'dist_op.shard_tensor',
                'pd_op.send_v2',
            ]
        else:
            np.testing.assert_equal(dist_program.num_ops(), 5)
            std_ops = [
                'builtin.parameter',
                'pd_op.data',
                'dist_op.shard_tensor',
                'pd_op.recv_v2',
                'pd_op.c_allreduce_sum_',
            ]
        np.testing.assert_equal(
            ops,
            std_ops,
        )
        for op in dist_program.global_block().ops:
            if op.name() == 'pd_op.send_v2':
                assert op.dist_attr.num_operands() == 1
                assert op.dist_attr.num_results() == 0
                op_operand_dist_attr = op.dist_attr.operand_dist_attr(0)

                assert op.dist_attr.process_mesh == self._in_mesh
                assert op_operand_dist_attr.process_mesh == self._in_mesh
                assert op_operand_dist_attr.dims_mapping == [-1, -1]
                assert op_operand_dist_attr.partial_status == {
                    0: paddle.distributed.ReduceType.kRedSum
                }

            elif op.name() == 'pd_op.recv_v2':
                # check op dist_attr
                assert op.dist_attr.num_operands() == 0
                assert op.dist_attr.num_results() == 1

                op_result_dist_attr = op.dist_attr.result_dist_attr(0)

                assert op_result_dist_attr.process_mesh == self._out_mesh
                assert op_result_dist_attr.dims_mapping == [-1, -1]
                assert op_result_dist_attr.partial_status == {
                    0: paddle.distributed.ReduceType.kRedSum
                }
            elif op.name() == 'pd_op.c_allreduce_sum_':
                continue
                # check op dist_attr
                assert op.dist_attr.num_operands() == 1
                assert op.dist_attr.num_results() == 1

                op_operand_dist_attr = op.dist_attr.operand_dist_attr(0)
                op_result_dist_attr = op.dist_attr.result_dist_attr(0)

                assert op.dist_attr.process_mesh == self._in_mesh
                assert op_operand_dist_attr.process_mesh == self._in_mesh
                assert op_operand_dist_attr.dims_mapping == [-1, -1]
                assert op_operand_dist_attr.partial_status == {
                    0: paddle.distributed.ReduceType.kRedSum
                }

                assert op_result_dist_attr.process_mesh == self._out_mesh
                assert op_result_dist_attr.dims_mapping == [-1, -1]
                assert op_result_dist_attr.partial_status == {}

                # check op_value dist_attr
                assert op.num_results() == 1
                op_value = op.result(0)
                assert op_value.is_dense_tensor_type()
                assert op_value.is_dist_dense_tensor_type()
                assert op_value.is_dist_dense_tensor_type()
                assert op_value.dist_attr().process_mesh == self._out_mesh
                assert op_value.dist_attr().dims_mapping == [-1, -1]
                assert op_value.dist_attr().partial_status == {}


if __name__ == '__main__':
    TestReshardSToRCrossMesh().run_pir_test_case()
