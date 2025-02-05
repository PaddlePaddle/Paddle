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

import logging

import paddle
import paddle.distributed as dist
from paddle.distributed.auto_parallel.static.process_group import (
    new_process_group,
)

from ..auto_parallel.static.utils import (
    get_logger,
)
from .pass_base import PassBase, register_pass

logger = get_logger(logging.INFO)


@register_pass("replace_with_parallel_cross_entropy")
class AutoParallelReplaceWithParallelCrossEntropyPass(PassBase):
    def __init__(self):
        super().__init__()
        hcg = dist.fleet.get_hybrid_communicate_group()
        self.model_parallel_group = hcg.get_model_parallel_group()
        self.tensor_parallel_degree = hcg.get_model_parallel_world_size()

    def _check_self(self):
        # The activation of this pass requires adopting a model parallel strategy.
        if self.tensor_parallel_degree < 2:
            return False
        return True

    def _check_conflict(self, other_pass):
        return True

    def _check_user(self, value):
        placement1 = value.placements
        for user in value.all_used_ops():
            for operand in user.operands_source():
                if operand.get_defining_op() != value.get_defining_op():
                    continue
                placement2 = operand.placements
                if placement1 != placement2:
                    return False
                break
        return True

    def _apply_single_impl(self, main_program, startup_program, context):
        del_ops = []
        new_ops = []

        for block in main_program.blocks:
            for op in reversed(block.ops):
                if op.name() == 'pd_op.cross_entropy_with_softmax':
                    operand1 = op.operand_source(0)
                    operand2 = op.operand_source(1)

                    # The `logit` input of the `cross_stropy_with_stoftmax` operator
                    # meed split along the column.
                    placement1 = operand1.placements
                    if not placement1[1].is_shard():
                        return

                    process_ids = operand1.dist_attr().process_mesh.process_ids
                    group = new_process_group(sorted(process_ids))
                    ring_id = group.id
                    nranks = group.nranks
                    rank = paddle.distributed.get_rank()

                    ignore_index = op.attrs()["ignore_index"]
                    paddle.pir.set_insertion_point(op)
                    softmax, loss = paddle._C_ops.c_softmax_with_cross_entropy(
                        operand1, operand2, ignore_index, ring_id, rank, nranks
                    )
                    op.result(0).replace_all_uses_with(softmax)
                    op.result(1).replace_all_uses_with(loss)
                    del_ops.append(op)
                    new_ops.append(softmax.get_defining_op())

        for op in del_ops:
            for result in op.results():
                assert result.use_empty()
            op.erase()
        # In the forward program, the placements of the newly added OP
        # output should be consistent with the placements of the user OP input
        for op in new_ops:
            for result in op.results():
                assert self._check_user(result)
        return
