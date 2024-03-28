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


def apply_partition_pass(program):
    new_program = program.clone()
    with paddle.static.program_guard(new_program):
        for op in new_program.global_block().ops:
            # assert len(op.operands()) == len(op.dist_attr().operand_dist_attrs()), f'The number of operand and operand_dist_attrs are not equal in op: {op}'
            for var, operand_dist_attr in zip(
                op.operands(), op.dist_attr().operand_dist_attrs()
            ):
                if (
                    var.source().is_dist_dense_tensor_type()
                    and var.source().dist_attr() != operand_dist_attr
                ):
                    paddle.pir.set_insertion_point(op)
                    # insert reshard
                    reshard_var = paddle._pir_ops.reshard_v2(
                        var.source(), operand_dist_attr
                    )
                    var.set_source(reshard_var)
    return new_program


def apply_reshard_pass(program):
    pass
