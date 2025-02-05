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
    ReshardPasses,
)
from paddle.distributed.auto_parallel.static.utils import set_all_ops_op_role
from paddle.distributed.fleet.meta_optimizers.common import OpRole


class TestReshardNdMesh:
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._seed = eval(os.getenv("seed"))
        self.BATCH_SIZE = 2
        self.SEQ_LEN = 4
        self.HIDDEN_SIZE = 8
        self._backend = os.getenv("backend")
        self._mesh = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=["x", "y"])

    def validate(
        self, op, tgt_operand, tgt_result, tgt_in_value, tgt_out_value
    ):
        # tgt_* are tuples, format: (process_ids, dims_mapping, partial_status)
        operand_dist_attr = op.dist_attr.operand(0).as_tensor_dist_attr()
        result_dist_attr = op.dist_attr.result(0).as_tensor_dist_attr()

        assert operand_dist_attr.process_mesh.process_ids == tgt_operand[0]
        assert operand_dist_attr.dims_mapping == tgt_operand[1]
        assert operand_dist_attr.partial_status == tgt_operand[2]

        assert result_dist_attr.process_mesh.process_ids == tgt_result[0]
        assert result_dist_attr.dims_mapping == tgt_result[1]
        assert result_dist_attr.partial_status == tgt_result[2]

        in_value = op.operand_source(0)
        out_value = op.result(0)
        assert in_value.is_dist_dense_tensor_type()
        assert out_value.is_dist_dense_tensor_type()
        assert in_value.dist_attr().process_mesh.process_ids == tgt_in_value[0]
        assert in_value.dist_attr().dims_mapping == tgt_in_value[1]
        assert in_value.dist_attr().partial_status == tgt_in_value[2]

        assert (
            out_value.dist_attr().process_mesh.process_ids == tgt_out_value[0]
        )
        assert out_value.dist_attr().dims_mapping == tgt_out_value[1]
        assert out_value.dist_attr().partial_status == tgt_out_value[2]

    def create_program(self, input_shape, input_placements, output_placements):
        paddle.enable_static()
        if self._backend == "cpu":
            paddle.set_device("cpu")
            place = paddle.CPUPlace()
        elif self._backend == "gpu":
            place = paddle.CUDAPlace(dist.get_rank())

        with paddle.pir_utils.IrGuard():
            main_program = paddle.base.Program()
            with paddle.base.program_guard(main_program):
                input = paddle.ones(name='input', shape=input_shape)
                dist_input = dist.shard_tensor(
                    input, self._mesh, input_placements
                )
                dist_out = paddle._C_ops.reshard(
                    dist_input, self._mesh, output_placements
                )
            dist_program = main_program.clone()
            set_all_ops_op_role(dist_program.global_block(), OpRole.Forward)
            ReshardPasses.apply_reshard_pass(dist_program)

        return main_program, dist_program

    def run_pp_to_rr_case(self):
        # [Partial(), Partial()] --> [Replicate(), Replicate()]
        # ops: all_reduce sum + all_reduce sum
        main_program, dist_program = self.create_program(
            [self.BATCH_SIZE, self.SEQ_LEN, self.HIDDEN_SIZE],
            [
                dist.Partial(dist.ReduceType.kRedSum),
                dist.Partial(dist.ReduceType.kRedSum),
            ],
            [dist.Replicate(), dist.Replicate()],
        )

        new_ops = dist_program.global_block().ops
        new_ops_name = [op.name() for op in dist_program.global_block().ops]

        rank_id = dist.get_rank()
        assert dist_program.global_block().ops[-2].name() == "pd_op.all_reduce"
        assert (
            dist_program.global_block().ops[-2].int_attr("reduce_type")
            == dist.ReduceOp.SUM
        )
        assert dist_program.global_block().ops[-1].name() == "pd_op.all_reduce"
        assert (
            dist_program.global_block().ops[-1].int_attr("reduce_type")
            == dist.ReduceOp.SUM
        )

        # check the first allreduce_sum
        op = new_ops[-2]
        if rank_id == 0 or rank_id == 2:
            process_ids = [0, 2]
        elif rank_id == 1 or rank_id == 3:
            process_ids = [1, 3]
        tgt_operand = (process_ids, [-1, -1, -1], {0: dist.ReduceType.kRedSum})
        tgt_result = (process_ids, [-1, -1, -1], {})
        tgt_in_value = (
            self._mesh.process_ids,
            [-1, -1, -1],
            {0: dist.ReduceType.kRedSum, 1: dist.ReduceType.kRedSum},
        )
        tgt_out_value = (
            self._mesh.process_ids,
            [-1, -1, -1],
            {1: dist.ReduceType.kRedSum},
        )
        self.validate(op, tgt_operand, tgt_result, tgt_in_value, tgt_out_value)

        # check the second allreduce_sum
        op = new_ops[-1]
        if rank_id == 0 or rank_id == 1:
            process_ids = [0, 1]
        elif rank_id == 2 or rank_id == 3:
            process_ids = [2, 3]
        tgt_operand = (process_ids, [-1, -1, -1], {0: dist.ReduceType.kRedSum})
        tgt_result = (process_ids, [-1, -1, -1], {})
        tgt_in_value = (
            self._mesh.process_ids,
            [-1, -1, -1],
            {1: dist.ReduceType.kRedSum},
        )
        tgt_out_value = (self._mesh.process_ids, [-1, -1, -1], {})
        self.validate(op, tgt_operand, tgt_result, tgt_in_value, tgt_out_value)

    def run_pr_to_rs_case(self):
        # [Partial(), Replicate()] --> [Replicate(), Shard(1)]
        # all_reduce sum + slice
        main_program, dist_program = self.create_program(
            [self.BATCH_SIZE, self.SEQ_LEN, self.HIDDEN_SIZE],
            [dist.Partial(dist.ReduceType.kRedSum), dist.Replicate()],
            [dist.Replicate(), dist.Shard(1)],
        )

        new_ops = dist_program.global_block().ops
        new_ops_name = [op.name() for op in dist_program.global_block().ops]

        check_all_reduce_sum = any(
            op.name() == "pd_op.all_reduce"
            and op.int_attr("reduce_type") == dist.ReduceOp.SUM
            for op in dist_program.global_block().ops
        )
        assert check_all_reduce_sum
        rank_id = dist.get_rank()
        assert new_ops_name[-1] == "pd_op.slice"

        # check the allreduce_sum
        op = new_ops[new_ops_name.index("pd_op.all_reduce")]
        if rank_id == 0 or rank_id == 2:
            process_ids = [0, 2]
        elif rank_id == 1 or rank_id == 3:
            process_ids = [1, 3]
        tgt_operand = (process_ids, [-1, -1, -1], {0: dist.ReduceType.kRedSum})
        tgt_result = (process_ids, [-1, -1, -1], {})
        tgt_in_value = (
            self._mesh.process_ids,
            [-1, -1, -1],
            {0: dist.ReduceType.kRedSum},
        )
        tgt_out_value = (self._mesh.process_ids, [-1, -1, -1], {})
        self.validate(op, tgt_operand, tgt_result, tgt_in_value, tgt_out_value)

        # check the second slice
        op = new_ops[-1]
        if rank_id == 0 or rank_id == 1:
            process_ids = [0, 1]
        elif rank_id == 2 or rank_id == 3:
            process_ids = [2, 3]
        tgt_operand = (process_ids, [-1, -1, -1], {})
        tgt_result = (process_ids, [-1, 0, -1], {})
        tgt_in_value = (self._mesh.process_ids, [-1, -1, -1], {})
        tgt_out_value = (self._mesh.process_ids, [-1, 1, -1], {})

    def run_pr_to_ss_case(self):
        self.create_program(
            [self.BATCH_SIZE, self.SEQ_LEN, self.HIDDEN_SIZE],
            [dist.Partial(dist.ReduceType.kRedSum), dist.Replicate()],
            [dist.Shard(0), dist.Shard(1)],
        )

    def run_ss_to_ss_case(self):
        # [Shard(0), Shard(1)] --> [Shard(1), Shard(0)]
        # all_gather+all_gather+slice+slice
        main_program, dist_program = self.create_program(
            [self.BATCH_SIZE, self.SEQ_LEN, self.HIDDEN_SIZE],
            [dist.Shard(0), dist.Shard(1)],
            [dist.Shard(1), dist.Shard(0)],
        )

        new_ops = dist_program.global_block().ops
        new_ops_name = [op.name() for op in dist_program.global_block().ops]

        all_gather_ops = []
        slice_ops = []
        for i, op in enumerate(new_ops):
            if op.name() == "pd_op.all_gather":
                all_gather_ops.append(op)
            elif op.name() == "pd_op.slice":
                slice_ops.append(op)
        assert len(all_gather_ops) == 2
        assert len(slice_ops) == 2
        rank_id = dist.get_rank()

        # check the first all_gather
        op = all_gather_ops[0]
        if rank_id == 0 or rank_id == 1:
            process_ids = [0, 1]
        elif rank_id == 2 or rank_id == 3:
            process_ids = [2, 3]
        tgt_operand = (
            process_ids,
            [-1, 0, -1],
            {},
        )  # process_ids, dims_mapping, partial_status
        tgt_result = (process_ids, [-1, -1, -1], {})
        tgt_in_value = (self._mesh.process_ids, [0, 1, -1], {})
        tgt_out_value = (self._mesh.process_ids, [0, -1, -1], {})
        self.validate(op, tgt_operand, tgt_result, tgt_in_value, tgt_out_value)

        # check the second all_gather
        op = all_gather_ops[1]
        if rank_id == 0 or rank_id == 2:
            process_ids = [0, 2]
        elif rank_id == 1 or rank_id == 3:
            process_ids = [1, 3]
        tgt_operand = (
            process_ids,
            [0, -1, -1],
            {},
        )  # process_ids, dims_mapping, partial_status
        tgt_result = (process_ids, [-1, -1, -1], {})
        tgt_in_value = (self._mesh.process_ids, [0, -1, -1], {})
        tgt_out_value = (self._mesh.process_ids, [-1, -1, -1], {})
        self.validate(op, tgt_operand, tgt_result, tgt_in_value, tgt_out_value)

        # check the first slice
        op = slice_ops[0]
        if rank_id == 0 or rank_id == 2:
            process_ids = [0, 2]
        elif rank_id == 1 or rank_id == 3:
            process_ids = [1, 3]
        tgt_operand = (process_ids, [-1, -1, -1], {})
        tgt_result = (process_ids, [-1, 0, -1], {})
        tgt_in_value = (self._mesh.process_ids, [-1, -1, -1], {})
        tgt_out_value = (self._mesh.process_ids, [-1, 0, -1], {})
        self.validate(op, tgt_operand, tgt_result, tgt_in_value, tgt_out_value)

        # check the second slice
        op = slice_ops[1]
        if rank_id == 0 or rank_id == 1:
            process_ids = [0, 1]
        elif rank_id == 2 or rank_id == 3:
            process_ids = [2, 3]
        tgt_operand = (process_ids, [-1, -1, -1], {})
        tgt_result = (process_ids, [0, -1, -1], {})
        tgt_in_value = (self._mesh.process_ids, [-1, 0, -1], {})
        tgt_out_value = (self._mesh.process_ids, [1, 0, -1], {})
        self.validate(op, tgt_operand, tgt_result, tgt_in_value, tgt_out_value)

    def run_ps_to_ps_case(self):
        # [Partial(), Shard(0)] --> [Replicate(), Shard(1)]
        # all_reduce sum + all_gather + slice
        main_program, dist_program = self.create_program(
            [self.BATCH_SIZE, self.SEQ_LEN, self.HIDDEN_SIZE],
            [dist.Partial(dist.ReduceType.kRedSum), dist.Shard(0)],
            [dist.Replicate(), dist.Shard(1)],
        )

        ops = dist_program.global_block().ops
        op_names = [op.name() for op in ops]
        check_all_reduce_sum = any(
            op.name() == "pd_op.all_reduce"
            and op.int_attr("reduce_type") == dist.ReduceOp.SUM
            for op in ops
        )
        assert check_all_reduce_sum
        assert "pd_op.all_gather" in op_names
        assert "pd_op.slice" in op_names

        allgather_op = ops[op_names.index("pd_op.all_gather")]
        allreduce_sum_op = ops[op_names.index("pd_op.all_reduce")]
        slice_op = ops[op_names.index("pd_op.slice")]

        # check the allgather
        rank_id = dist.get_rank()
        if rank_id in [0, 1]:
            process_ids = [0, 1]
        elif rank_id in [2, 3]:
            process_ids = [2, 3]
        tgt_operand = (process_ids, [0, -1, -1], {})
        tgt_result = (process_ids, [-1, -1, -1], {})
        tgt_in_value = (
            self._mesh.process_ids,
            [1, -1, -1],
            {0: dist.ReduceType.kRedSum},
        )
        tgt_out_value = (
            self._mesh.process_ids,
            [-1, -1, -1],
            {0: dist.ReduceType.kRedSum},
        )
        self.validate(
            allgather_op, tgt_operand, tgt_result, tgt_in_value, tgt_out_value
        )

        # check the allreduce_sum
        if rank_id in [0, 2]:
            process_ids = [0, 2]
        elif rank_id in [1, 3]:
            process_ids = [1, 3]
        tgt_operand = (process_ids, [-1, -1, -1], {0: dist.ReduceType.kRedSum})
        tgt_result = (process_ids, [-1, -1, -1], {})
        tgt_in_value = (
            self._mesh.process_ids,
            [-1, -1, -1],
            {0: dist.ReduceType.kRedSum},
        )
        tgt_out_value = (self._mesh.process_ids, [-1, -1, -1], {})
        self.validate(
            allreduce_sum_op,
            tgt_operand,
            tgt_result,
            tgt_in_value,
            tgt_out_value,
        )

        # check the slice
        if rank_id in [0, 1]:
            process_ids = [0, 1]
        elif rank_id in [2, 3]:
            process_ids = [2, 3]
        tgt_operand = (process_ids, [-1, -1, -1], {})
        tgt_result = (process_ids, [-1, 0, -1], {})
        tgt_in_value = (self._mesh.process_ids, [-1, -1, -1], {})
        tgt_out_value = (self._mesh.process_ids, [-1, 1, -1], {})
        self.validate(
            slice_op, tgt_operand, tgt_result, tgt_in_value, tgt_out_value
        )

    def run_test_cases(self):
        self.run_pp_to_rr_case()
        self.run_pr_to_rs_case()
        self.run_pr_to_ss_case()
        self.run_ss_to_ss_case()
        self.run_ps_to_ps_case()


if __name__ == '__main__':
    TestReshardNdMesh().run_test_cases()
