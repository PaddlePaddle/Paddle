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

import paddle
import paddle.distributed as dist
from paddle.distributed.auto_parallel.static.pir_pass import (
    apply_reshard_pass,
)


class TestReshardNdMesh:
    def __init__(self):
        # self._shape = eval(os.getenv("shape"))
        # self._dtype = os.getenv("dtype")
        # self._seeds = eval(os.getenv("seeds"))
        # self._shard = eval(os.getenv("shard"))
        self._backend = os.getenv("backend")
        self._mesh = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=["x", "y"])

    def run_pp_to_rr_case(self):
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
                input = paddle.ones(
                    name='input', shape=[BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE]
                )
                dist_input = dist.shard_tensor(
                    input,
                    self._mesh,
                    [
                        dist.Partial(dist.ReduceType.kRedSum),
                        dist.Partial(dist.ReduceType.kRedSum),
                    ],
                )
                dist_out = paddle._C_ops.reshard(
                    dist_input, self._mesh, [dist.Replicate(), dist.Replicate()]
                )
            dist_program = main_program.clone()
            apply_reshard_pass(dist_program)
            new_ops = dist_program.global_block().ops
            old_ops_name = [op.name() for op in main_program.global_block().ops]
            new_ops_name = [op.name() for op in dist_program.global_block().ops]

            rank_id = dist.get_rank()
            assert new_ops_name[-2] == "pd_op.c_allreduce_sum_"
            assert new_ops_name[-1] == "pd_op.c_allreduce_sum_"

            # check the first allreduce_sum
            op = new_ops[-2]
            in_dist_attr = op.dist_attr.operand(0).as_tensor_dist_attr()
            out_dist_attr = op.dist_attr.result(0).as_tensor_dist_attr()
            assert in_dist_attr.partial_status == {0: dist.ReduceType.kRedSum}
            assert in_dist_attr.dims_mapping == [-1, -1, -1]
            assert out_dist_attr.partial_status == {}
            assert out_dist_attr.dims_mapping == [-1, -1, -1]
            if rank_id == 0 or rank_id == 2:
                assert in_dist_attr.process_mesh.process_ids == [0, 2]
            elif rank_id == 1 or rank_id == 3:
                assert in_dist_attr.process_mesh.process_ids == [1, 3]

            in_value = op.operand_source(0)
            out_value = op.result(0)
            assert in_value.is_dist_dense_tensor_type()
            assert out_value.is_dist_dense_tensor_type()
            assert in_value.dist_attr().dims_mapping == [-1, -1, -1]
            assert in_value.dist_attr().process_mesh == self._mesh
            assert in_value.dist_attr().partial_status == {
                0: dist.ReduceType.kRedSum,
                1: dist.ReduceType.kRedSum,
            }
            assert out_value.dist_attr().dims_mapping == [-1, -1, -1]
            assert out_value.dist_attr().process_mesh == self._mesh
            assert out_value.dist_attr().partial_status == {
                1: dist.ReduceType.kRedSum
            }

            # check the second allreduce_sum
            op = new_ops[-1]
            in_dist_attr = op.dist_attr.operand(0).as_tensor_dist_attr()
            out_dist_attr = op.dist_attr.result(0).as_tensor_dist_attr()
            assert in_dist_attr.partial_status == {0: dist.ReduceType.kRedSum}
            assert in_dist_attr.dims_mapping == [-1, -1, -1]
            assert out_dist_attr.partial_status == {}
            assert out_dist_attr.dims_mapping == [-1, -1, -1]
            if rank_id == 0 or rank_id == 1:
                assert in_dist_attr.process_mesh.process_ids == [0, 1]
            elif rank_id == 2 or rank_id == 3:
                assert in_dist_attr.process_mesh.process_ids == [2, 3]

            in_value = op.operand_source(0)
            out_value = op.result(0)
            assert in_value.is_dist_dense_tensor_type()
            assert out_value.is_dist_dense_tensor_type()
            assert in_value.dist_attr().dims_mapping == [-1, -1, -1]
            assert in_value.dist_attr().process_mesh == self._mesh
            assert in_value.dist_attr().partial_status == {
                1: dist.ReduceType.kRedSum
            }
            assert out_value.dist_attr().dims_mapping == [-1, -1, -1]
            assert out_value.dist_attr().process_mesh == self._mesh
            assert out_value.dist_attr().partial_status == {}


if __name__ == '__main__':
    TestReshardNdMesh().run_pp_to_rr_case()
