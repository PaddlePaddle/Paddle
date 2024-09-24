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

import re
import warnings

import paddle
import paddle.distributed as dist
from paddle.distributed.auto_parallel.static.converter import Converter
from paddle.distributed.fleet.meta_optimizers.common import OpRole
from paddle.static import global_scope

from .pass_base import PassBase, register_pass


@register_pass("auto_parallel_c_embedding_pass")
class AutoParallelCEmbeddingPass(PassBase):
    def __init__(self):
        super().__init__()

    def _check_self(self):
        main_program = self.get_attr("dist_program")
        ops = main_program.global_block().ops
        for i, op in enumerate(ops):
            if op.name() == 'pd_op.embedding':
                placements = op.operand(1).source().placements
                dim_map, partial_status = (
                    dist.auto_parallel.placement_type.to_dim_map(
                        placements, op.operand(1).source().ndim
                    )
                )
                if dim_map[1] == -1 or dim_map[1] in partial_status:
                    warnings.warn(
                        "The c_embedding pass is only applicable to column-wise parallel `embedding` kernel."
                    )
                    return False
                else:
                    return True
        warnings.warn(
            "The c_embedding pass is only applicable to column-wise parallel `embedding` kernel."
        )
        return False

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, context):
        concrete_program = self.get_attr("concrete_program")
        ops = main_program.global_block().ops
        for i, op in enumerate(ops):
            if op.name() == 'pd_op.embedding':
                # weight
                placements = op.operand(1).source().placements
                dim_map, partial_status = (
                    dist.auto_parallel.placement_type.to_dim_map(
                        placements, op.operand(1).source().ndim
                    )
                )
                mp_axis = dim_map[1]
                dim_map = [mp_axis, -1]
                dist_attr_w = (
                    paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                        op.operand(1).source().process_mesh,
                        dim_map,
                        partial_status,
                    )
                )
                dist_type_input0 = paddle.base.libpaddle.pir.cvt_to_dist_type(
                    op.operand(1).source().type(), dist_attr_w
                )
                op.operand(1).source().set_type(dist_type_input0)

                # update c_emebedding weight dynamic parameters
                dy_params = concrete_program.parameters[0]
                pattern = re.compile(r'embedding_.*\.w_0\.dist')
                for index, param in enumerate(dy_params):
                    if pattern.match(param.name):
                        dist_attr = {
                            "dims_mapping": dist_attr_w.dims_mapping,
                            "process_shape": dist_attr_w.process_mesh.shape,
                            "process_group": dist_attr_w.process_mesh.process_ids,
                        }
                        place = paddle.framework.CUDAPlace(
                            paddle.distributed.ParallelEnv().dev_id
                        )
                        scope_param = (
                            global_scope().var(param.name).get_tensor()
                        )
                        scope_param._share_data_with(
                            param.get_tensor().get_tensor()
                        )
                        sliced_param = Converter.slice_with_dist_attr(
                            param.numpy(), dist_attr
                        )
                        scope_param.set(sliced_param, place)
                        param.get_tensor()._clear()
                        concrete_program.parameters[0][index] = None

                # replace embedding with c_embedding
                paddle.pir.set_insertion_point(op)
                num_embeddings = op.operand(1).source().type().shape[0]
                world_size = paddle.distributed.get_world_size()
                rank = paddle.distributed.get_rank()
                per_part_size = num_embeddings // world_size
                vocab_start_index = rank * per_part_size
                t_op = paddle._C_ops.c_embedding(
                    op.operand(1).source(),
                    op.operand(0).source(),
                    vocab_start_index,
                    num_embeddings,
                )
                t_op.get_defining_op().op_role = int(OpRole.Forward)
                new_op = t_op.get_defining_op()
                op.result(0).replace_all_uses_with(t_op)
                op.erase()

                # update dims_mapping before c_embedding
                stack = [new_op.operand(0).source().get_defining_op()]
                while stack:
                    op = stack.pop()
                    if op.dist_attr is None:
                        continue
                    change = False
                    operands, results = [], []
                    if op.num_operands() > 0:
                        for operand, operand_dist in zip(
                            op.operands_source(), op.dist_attr.operands()
                        ):
                            placements = operand.placements
                            placements_dist = (
                                operand_dist.as_tensor_dist_attr().placements
                            )
                            if placements != placements_dist:
                                dim_map, partial_status = (
                                    dist.auto_parallel.placement_type.to_dim_map(
                                        placements, operand.ndim
                                    )
                                )
                                dist_attr_new = paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                                    operand.process_mesh,
                                    dim_map,
                                    partial_status,
                                )
                                dist_type = (
                                    paddle.base.libpaddle.pir.cvt_to_dist_type(
                                        operand.type(), dist_attr_new
                                    )
                                )
                                operand.set_type(dist_type)
                                operands.append(dist_attr_new)
                                change = True
                                stack.append(operand.get_defining_op())
                            else:
                                operands.append(
                                    operand_dist.as_tensor_dist_attr()
                                )
                    if op.num_results() > 0:
                        for result, result_dist in zip(
                            op.results(), op.dist_attr.results()
                        ):
                            placements = result.placements
                            placements_dist = (
                                result_dist.as_tensor_dist_attr().placements
                            )
                            if placements != placements_dist:
                                dim_map, partial_status = (
                                    dist.auto_parallel.placement_type.to_dim_map(
                                        placements, result.ndim
                                    )
                                )
                                dist_attr_new = paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                                    result.process_mesh,
                                    dim_map,
                                    partial_status,
                                )
                                dist_type = (
                                    paddle.base.libpaddle.pir.cvt_to_dist_type(
                                        result.type(), dist_attr_new
                                    )
                                )
                                result.set_type(dist_type)
                                results.append(dist_attr_new)
                                change = True

                            else:
                                results.append(
                                    result_dist.as_tensor_dist_attr()
                                )
                    if change:
                        process_mesh = (
                            op.results()[0].process_mesh
                            if op.num_results() > 0
                            else op.operand(0).source().process_mesh
                        )
                        op.dist_attr = (
                            paddle.base.libpaddle.pir.create_op_dist_attribute(
                                process_mesh,
                                operands,
                                results,
                            )
                        )

                # update dims_mapping after c_embedding
                stack = list(new_op.results()[0].all_used_ops())
                while stack:
                    op = stack.pop()
                    if op.dist_attr is None:
                        continue
                    change = False
                    operands, results = [], []
                    if op.num_operands() > 0:
                        for operand, operand_dist in zip(
                            op.operands_source(), op.dist_attr.operands()
                        ):
                            placements = operand.placements
                            placements_dist = (
                                operand_dist.as_tensor_dist_attr().placements
                            )
                            if placements != placements_dist:
                                dim_map, partial_status = (
                                    dist.auto_parallel.placement_type.to_dim_map(
                                        placements, operand.ndim
                                    )
                                )
                                dist_attr_new = paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                                    operand.process_mesh,
                                    dim_map,
                                    partial_status,
                                )
                                dist_type = (
                                    paddle.base.libpaddle.pir.cvt_to_dist_type(
                                        operand.type(), dist_attr_new
                                    )
                                )
                                operand.set_type(dist_type)
                                operands.append(dist_attr_new)
                                change = True
                            else:
                                operands.append(
                                    operand_dist.as_tensor_dist_attr()
                                )
                    if op.num_results() > 0:
                        for result, result_dist in zip(
                            op.results(), op.dist_attr.results()
                        ):
                            placements = result.placements
                            placements_dist = (
                                result_dist.as_tensor_dist_attr().placements
                            )
                            if placements != placements_dist:
                                dim_map, partial_status = (
                                    dist.auto_parallel.placement_type.to_dim_map(
                                        placements, result.ndim
                                    )
                                )
                                dist_attr_new = paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                                    result.process_mesh,
                                    dim_map,
                                    partial_status,
                                )
                                dist_type = (
                                    paddle.base.libpaddle.pir.cvt_to_dist_type(
                                        result.type(), dist_attr_new
                                    )
                                )
                                result.set_type(dist_type)
                                results.append(dist_attr_new)
                                change = True
                                for next_op in result.all_used_ops():
                                    stack.append(next_op)
                            else:
                                results.append(
                                    result_dist.as_tensor_dist_attr()
                                )

                    if change:
                        process_mesh = (
                            op.results()[0].process_mesh
                            if op.num_results() > 0
                            else op.operand(0).source().process_mesh
                        )
                        op.dist_attr = (
                            paddle.base.libpaddle.pir.create_op_dist_attribute(
                                process_mesh,
                                operands,
                                results,
                            )
                        )
