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
from .base_reshard_func import ReshardFunction, is_partial, is_replicated
from .same_status_reshard_func import SameStatusReshardFunction


class PToRReshardFunction(ReshardFunction):
    def is_suitable(self, src_dist_attr, dst_dist_attr):
        if not is_partial(src_dist_attr):
            return False

        if not is_replicated(dst_dist_attr):
            return False

        in_mesh = src_dist_attr.process_mesh
        out_mesh = dst_dist_attr.process_mesh

        if in_mesh.ndim != 1:
            return False
        if out_mesh.ndim != 1:
            return False
        if in_mesh != out_mesh:
            return False
        return True

    def reshard(
        self, program, op, src_dist_attr, dst_dist_attr, reshard_op=True
    ):
        src_mesh = src_dist_attr.process_mesh
        src_reduce_type = src_dist_attr.partial_status[0]
        reduce_mean = False
        if src_reduce_type == ReduceOp.AVG:
            src_reduce_type = ReduceOp.SUM
            reduce_mean = True

        op_value = op.result(0)
        op_type = op_value.type()
        if reshard_op:
            paddle.pir.set_insertion_point(op)
            op_value = op.operand_source(0)
        else:
            paddle.pir.set_insertion_point_after(op)
        group = new_process_group(src_mesh.process_ids)
        reduced_value = paddle._pir_ops.c_allreduce_sum_(
            op_value, group.id, False, False
        )

        # set dist type and dist attr
        reduced_value.set_type(op_type)
        reduced_value.get_defining_op().dist_attr = (
            paddle.base.libpaddle.pir.create_op_dist_attribute(
                src_mesh, [src_dist_attr], [dst_dist_attr]
            )
        )
        if reshard_op:
            op.result(0).replace_all_uses_with(reduced_value)
            program.global_block().remove_op(op)


class PToRReshardFunctionCrossMesh(ReshardFunction):
    def is_suitable(self, src_dist_attr, dst_dist_attr):
        if not is_partial(src_dist_attr):
            return False

        if not is_replicated(dst_dist_attr):
            return False

        in_mesh = src_dist_attr.process_mesh
        out_mesh = dst_dist_attr.process_mesh

        if (
            in_mesh.ndim != 1
            or out_mesh.ndim != 1
            or in_mesh.shape != out_mesh.shape
        ):
            return False

        if in_mesh == out_mesh:
            return False

        return True

    def reshard(self, program, op, src_dist_attr, dst_dist_attr):
        same_status_func = SameStatusReshardFunction()
        tmp_dist_attr = paddle.base.libpaddle.pir.create_tensor_dist_attribute(
            dst_dist_attr.process_mesh,
            src_dist_attr.dims_mapping,
            src_dist_attr.partial_status,
        )
        pre_op, out_dist_attr = same_status_func.reshard(
            program, op, src_dist_attr, tmp_dist_attr
        )

        if pre_op is None:
            return None, out_dist_attr

        curr_global_rank = paddle.distributed.get_rank()
        if curr_global_rank in dst_dist_attr.process_mesh.process_ids:
            p_to_r_func = PToRReshardFunction()
            assert p_to_r_func.is_suitable(
                out_dist_attr, dst_dist_attr
            ), f"Invoke the p to r reshard function is not valid from {pre_op.dist_attr()} to {dst_dist_attr}"
            p_to_r_func.reshard(
                program, pre_op, out_dist_attr, dst_dist_attr, False
            )
