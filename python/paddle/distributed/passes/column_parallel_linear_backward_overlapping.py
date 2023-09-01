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

import collections

from .pass_base import PassBase, register_pass
from .pass_utils import AutoParallelStreamType


# For allreduce pattern in the backward phase of column parallel linear:
#   dX, dY = matmul_grad(X, Y, dOut)
#   dX = c_allreduce_sum(dX)
# Split matmul_grad to 2 matmul:
#   dX = mutmul(dOut, Y^T)
#   dX = c_allreduce_sum(dX)
#   dY = matmul(X^T, dOut)
#
# Then the c_allreduce_sum can overlap with the compute of dY.
@register_pass("column_parallel_linear_backward_overlapping")
class ColumnParallelLinearBackwardOverlappingPass(PassBase):
    def __init__(self):
        super().__init__()
        self.set_attr("allreduce_stream", None)

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, context):
        block = main_program.global_block()
        matmul_grad_id_to_allreduce_id = (
            self._get_all_matmul_grad_and_allreduce_pairs(block)
        )
        self._split_matmul_grad_and_multi_streaming_allreduce(
            block, matmul_grad_id_to_allreduce_id
        )

    def _get_all_matmul_grad_and_allreduce_pairs(self, block):
        ops = block.ops
        op_num = len(ops)
        matmul_grad_id_to_allreduce_id = collections.OrderedDict()
        for i, op_i in enumerate(ops):
            if (
                op_i.type == 'matmul_v2_grad'
                and op_i.attr("trans_x") is False
                and op_i.attr("trans_y") is False
            ):
                x_grad = op_i.output("X@GRAD")
                for j in range(i + 1, op_num):
                    op_j = ops[j]
                    if (
                        op_j.type == 'c_allreduce_sum'
                        and op_j.input("X") == x_grad
                    ):
                        matmul_grad_id_to_allreduce_id[i] = j
        return matmul_grad_id_to_allreduce_id

    def _split_matmul_grad_and_multi_streaming_allreduce(
        self, block, matmul_grad_id_to_allreduce_id
    ):
        ops = block.ops

        for matmul_grad_id, allreduce_id in reversed(
            matmul_grad_id_to_allreduce_id.items()
        ):
            matmul_grad_op = ops[matmul_grad_id]
            allreduce_op = ops[allreduce_id]

            tran_x = matmul_grad_op.attr("trans_x")
            assert (
                not tran_x
            ), f"matmul_grad(id={matmul_grad_id}) with tran_x == True is not supported for column parallel linear backward overlapping"
            tran_y = matmul_grad_op.attr("trans_y")
            assert (
                not tran_y
            ), f"matmul_grad(id={matmul_grad_id}) with tran_y == True is not supported for column parallel linear backward overlapping"

            allreduce_op.dist_attr.execution_stream = (
                AutoParallelStreamType.MP_STREAM.value
            )

            x = matmul_grad_op.input("X")
            y = matmul_grad_op.input("Y")
            out_grad = matmul_grad_op.input("Out@GRAD")
            x_grad = matmul_grad_op.output("X@GRAD")
            y_grad = matmul_grad_op.output("Y@GRAD")
            op_role = matmul_grad_op.attr("op_role")

            # NOTE(Ruibiao): Required OP scheduling order: mutmul(dOut, Y^T) -> c_allreduce_sum(dX) -> matmul(X^T, dOut).
            # c_allreduce_sum(dX) and matmul(X^T, dOut) cannot be swapped. Otherwise, after buffer_shared_inplace_pass
            # adding share_buffer OP before c_allreduce_sum, c_allreduce_sum will synchronous with comp-stream, and then
            # the matmul op before it cannot be overlapped.
            block._insert_op_without_sync(
                allreduce_id + 1,
                type="matmul_v2",
                inputs={"X": x, "Y": out_grad},
                outputs={"Out": y_grad},
                attrs={"trans_x": True, "trans_y": False, "op_role": op_role},
            )
            block._insert_op_without_sync(
                matmul_grad_id + 1,
                type="matmul_v2",
                inputs={"X": out_grad, "Y": y},
                outputs={"Out": x_grad},
                attrs={"trans_x": False, "trans_y": True, "op_role": op_role},
            )
            block._remove_op(matmul_grad_id)
            block._sync_with_cpp()
