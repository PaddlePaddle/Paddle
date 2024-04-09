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

import paddle
from paddle.distributed.communication.reduce import ReduceOp

from ..process_group import new_process_group
from .base_reshard_func import ReshardFunction, register_reshard_func


class PToRReshardFunction(ReshardFunction):
    def is_suitable(self, src_dist_attr, dst_dist_attr):
        if not src_dist_attr.is_partial():
            return False
        if not dst_dist_attr.is_replicated():
            return False
        in_mesh = src_dist_attr.process_mesh()
        out_mesh = dst_dist_attr.process_mesh()

        if in_mesh.ndim != 1:
            return False
        if out_mesh.ndim != 1:
            return False
        if in_mesh != out_mesh:
            return False
        return True

    def eval(self, program, op, src_dist_attr, dst_dist_attr):
        src_mesh = src_dist_attr.process_mesh()
        src_process_ids = src_dist_attr.process_ids()
        src_partial_status = src_dist_attr.partial_status()
        src_reduce_type = src_partial_status(0)
        reduce_mean = False
        if src_reduce_type == ReduceOp.AVG:
            src_reduce_type = ReduceOp.SUM
            reduce_mean = True

        paddle.pir.set_insertion_point(op)
        group = new_process_group(src_process_ids)
        reduced_value = paddle._pir_ops.c_allreduce_sum_(
            op.operand(0).source(), group.id, False, False
        )
        reduced_value.set_type(op.result(0).type())
        op.result(0).replace_all_uses_with(reduced_value)
        program.global_block().remove_op(op)


register_reshard_func(PToRReshardFunction)
