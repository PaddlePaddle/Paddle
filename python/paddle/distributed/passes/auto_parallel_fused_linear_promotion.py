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


import logging

import numpy as np

from paddle.distributed.auto_parallel.static.utils import (
    is_optimize_op,
    is_recompute_op,
    naive_set_dist_op_attr_for_program_by_mesh_and_mapping,
    set_var_dist_attr,
)
from paddle.utils import unique_name

from ..utils.log_utils import get_logger
from .auto_parallel_sharding import (
    _inference_data_parallel_group_for_operator,
    _is_reshard_op,
    _skip_ops,
    is_forward_op,
)
from .pass_base import PassBase, register_pass

logger = get_logger(logging.INFO, "FusedLinearPromotionPass")

_supported_optimizer_type = [
    "adam",
    "adamax",
    "adamw",
    "decayed_adagrad",
    "momentum",
    "dgc_momentum",
    "lars_momentum",
    "merged_momentum",
    "lamb",
    "sgd",
]

FUSED_LINEAR_SOURCE_PATTERNS_LIST = [
    # amp_level == 'o2' or 'o3'
    {  # only MP
        "forward": ["matmul_v2", "c_allreduce_sum", "elementwise_add"],
        "backward": ["elementwise_add_grad", "matmul_v2_grad"],
    },
    {  # MP + SP
        "forward": ["matmul_v2", "reduce_scatter", "elementwise_add"],
        "backward": [
            "elementwise_add_grad",
            "c_allreduce_sum",
            "scale",
            "all_gather",
            "matmul_v2_grad",
            "all_gather",
        ],
    },
    {  # DP + MP
        "forward": ["matmul_v2", "c_allreduce_sum", "elementwise_add"],
        "backward": [
            "elementwise_add_grad",
            "c_allreduce_sum",
            "scale",
            "matmul_v2_grad",
        ],
    },
    {  # DP + MP + SP
        "forward": ["matmul_v2", "reduce_scatter", "elementwise_add"],
        "backward": [
            "elementwise_add_grad",
            "c_allreduce_sum",
            "scale",
            "c_allreduce_sum",
            "scale",
            "all_gather",
            "matmul_v2_grad",
            "all_gather",
        ],
    },
    # amp_level == 'o1'
    {
        "forward": ["matmul_v2", "c_allreduce_sum", "cast", "elementwise_add"],
        "backward": ["elementwise_add_grad", "matmul_v2_grad"],
    },
    {
        "forward": ["matmul_v2", "reduce_scatter", "cast", "elementwise_add"],
        "backward": [
            "elementwise_add_grad",
            "c_allreduce_sum",
            "scale",
            "all_gather",
            "all_gather",
            "matmul_v2_grad",
        ],
    },
    {
        "forward": ["matmul_v2", "c_allreduce_sum", "cast", "elementwise_add"],
        "backward": [
            "elementwise_add_grad",
            "c_allreduce_sum",
            "scale",
            "matmul_v2_grad",
        ],
    },
    {
        "forward": ["matmul_v2", "reduce_scatter", "cast", "elementwise_add"],
        "backward": [
            "elementwise_add_grad",
            "c_allreduce_sum",
            "scale",
            "c_allreduce_sum",
            "scale",
            "all_gather",
            "matmul_v2_grad",
            "all_gather",
        ],
    },
]


@register_pass("auto_parallel_fused_linear_promotion")
class FusedLinearPromotionPass(PassBase):
    """
    Apply pre-promotion that specialized for fused_linear_pass in tensor parallelism or sequence parallelism in Auto Parallel.
    """

    def __init__(self):
        super().__init__()
        self.set_attr("dist_context", None)
        self.set_attr("global_rank", -1)
        self.set_attr("enable_sp", False)
        self.set_attr("amp_level", "o0")
        self.set_attr("params_grads", None)

    def _check_self(self):
        if self.get_attr("dist_context") is None:
            return False
        if (not isinstance(self.get_attr("global_rank"), int)) or self.get_attr(
            "global_rank"
        ) < 0:
            return False
        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, context):
        self._dist_context = self.get_attr("dist_context")
        self._global_rank = int(self.get_attr("global_rank"))
        self._params_grads = self.get_attr("params_grads")
        self._amp_level = self.get_attr("amp_level")
        self._enable_sp = self.get_attr("enable_sp")
        self._is_amp_o1 = self._amp_level == 'o1'
        self._source_patterns = {}
        self._enable_dp, self._enable_mp = self._is_enable_dp_mp(
            self._dist_context
        )

        pattern_offset = 4 if self._is_amp_o1 else 0
        if self._enable_sp:
            if self._enable_dp:
                self._source_patterns = FUSED_LINEAR_SOURCE_PATTERNS_LIST[
                    3 + pattern_offset
                ]
            else:
                self._source_patterns = FUSED_LINEAR_SOURCE_PATTERNS_LIST[
                    1 + pattern_offset
                ]
        elif self._enable_mp:
            if self._enable_dp:
                self._source_patterns = FUSED_LINEAR_SOURCE_PATTERNS_LIST[
                    2 + pattern_offset
                ]
            else:
                self._source_patterns = FUSED_LINEAR_SOURCE_PATTERNS_LIST[
                    0 + pattern_offset
                ]
        else:
            logger.warning("Neither of sp and mp is enabled, skip this pass")
            return
        dp_group = None
        if self._enable_dp:
            dp_group = self._collective_data_parallel_groups(
                main_program.global_block()
            )

        # 1. get whether the current rank is first rank in mp
        self._is_first_rank = self._is_tp_sp_first_rank(
            self._dist_context, self._global_rank
        )
        logger.debug(f"before main_program: {main_program}")
        # 2. get the forward and backward op list indexes in source patterns
        (
            forward_segments,
            backward_segments,
        ) = self._get_forward_backward_op_segments(main_program)
        if len(forward_segments) == 0 or len(backward_segments) == 0:
            logger.warning(
                "No forward and backward op segments, skip this pass"
            )
            return
        # 3 transform the forward ops
        rename_var_names_map, deleted_bias_names = self._transform_forward(
            main_program,
            forward_segments,
            backward_segments,
            self._is_first_rank,
            self._enable_sp,
            self._is_amp_o1,
        )

        # 4 transform the backward ops
        self._transform_backward(
            main_program,
            backward_segments,
            rename_var_names_map,
            self._is_first_rank,
            self._enable_sp,
        )

        # 5. transform the optimizer ops
        self._transform_opt(
            main_program,
            deleted_bias_names,
            self._params_grads,
            self._is_first_rank,
            self._is_amp_o1,
        )
        logger.info(f"deleted_bias_names: {deleted_bias_names}")
        logger.debug(f"after main_program: {main_program}")

        # 6. transform the startup program
        self._transform_startup_program(
            startup_program, deleted_bias_names, dp_group, self._is_first_rank
        )

    def _is_tp_sp_first_rank(self, dist_context, rank):
        for process_mesh in dist_context.process_meshes:
            inner_mesh_shape = process_mesh.shape
            inner_mesh = (np.array(process_mesh.process_ids)).reshape(
                inner_mesh_shape
            )
            if len(inner_mesh_shape) == 1:
                return rank == min(process_mesh.process_ids)
            elif len(inner_mesh.shape) == 2:
                for id0 in range(inner_mesh_shape[0]):
                    if rank == min(inner_mesh[id0, :]):
                        return True
            elif len(inner_mesh.shape) == 3:
                for id0 in range(inner_mesh_shape[0]):
                    for id1 in range(inner_mesh_shape[1]):
                        if rank == min(inner_mesh[id0, id1, :]):
                            return True
            else:
                raise ValueError("inner mesh shape is not supported")
        return False

    def _is_enable_dp_mp(self, dist_context):
        for process_mesh in dist_context.process_meshes:
            inner_mesh_shape = process_mesh.shape
            inner_mesh = (np.array(process_mesh.process_ids)).reshape(
                inner_mesh_shape
            )
            if len(inner_mesh_shape) == 1:
                return False, inner_mesh_shape[0] > 1
            else:
                # DP * MP
                return inner_mesh_shape[-2] > 1, inner_mesh_shape[-1] > 1
        return False, False

    def _get_forward_backward_op_segments(self, main_program):
        """
        Get the operator segments according to the source patterns.
        """

        def can_match_pattern(
            ops, start_id, pattern, forward_matmul_inputs, is_backward=False
        ):
            """
            Check whether the ops in the range [start_id, start_id + len(pattern)] can match the pattern.
            If the ops is in forward pass, check it directly. However, when the ops is in backward pass,
            we need to additionally check whether the input of the last op in pattern is in forward_matmul_inputs to
            deal the case of enabling recompute.
            """
            new_id = start_id
            if not is_backward:
                for op_name in pattern:
                    if ops[new_id].type != op_name:
                        return False
                    new_id += 1
                forward_matmul_inputs.extend(ops[start_id].input_arg_names)
                return True
            else:
                for op_name in pattern:
                    if ops[new_id].type != op_name:
                        return False
                    new_id += 1
                matmul_grad_input_names = ops[new_id - 1].input_arg_names
                # for refined-recompute
                if (
                    matmul_grad_input_names[1] not in forward_matmul_inputs
                    and matmul_grad_input_names[2] not in forward_matmul_inputs
                ):
                    return False
                return True

        global_block = main_program.global_block()
        forward_segments = []
        backward_segments = []
        ops_len = len(global_block.ops)

        self._forward_patterns_len = len(self._source_patterns["forward"])
        self._backward_patterns_len = len(self._source_patterns["backward"])
        forward_matmul_inputs = []
        for id, op in enumerate(global_block.ops):
            if id > ops_len - self._backward_patterns_len:
                break
            if int(op.desc.attr('op_role')) == 0 or (
                is_recompute_op(op) and not op.type.endswith("_grad")
            ):  # forward
                if can_match_pattern(
                    global_block.ops,
                    id,
                    self._source_patterns["forward"],
                    forward_matmul_inputs,
                    is_backward=False,
                ):
                    forward_segments.append(
                        [id, id + self._forward_patterns_len]
                    )
            elif int(op.desc.attr('op_role')) == 1:  # backward
                if can_match_pattern(
                    global_block.ops,
                    id,
                    self._source_patterns["backward"],
                    forward_matmul_inputs,
                    is_backward=True,
                ):
                    backward_segments.append(
                        [id, id + self._backward_patterns_len]
                    )
            else:
                pass
        assert len(forward_segments) >= len(
            backward_segments
        ), "The number of forward segments should be not shorter than the number of backward segments."
        logger.info(f"forward_segments: {forward_segments}")
        logger.info(f"backward_segments: {backward_segments}")
        return forward_segments, backward_segments

    def _collective_data_parallel_groups(self, main_block):
        for op in main_block.ops:
            if not is_forward_op(op) or op.type in _skip_ops:
                continue
            # NOTE: there aren't dist_attr in the ops which reshard insert,
            # and should be skip in sharding.
            if _is_reshard_op(op):
                continue
            group = _inference_data_parallel_group_for_operator(
                self._global_rank, op, self._dist_context
            )
            if group is not None:
                return group
        return None

    def _transform_forward(
        self,
        main_program,
        forward_segments,
        backward_segments,
        is_first_rank,
        is_sp,
        is_amp_o1,
    ):
        """
        Transform the forward pass.
        """

        def _transform_forward_segment(
            global_block,
            forward_segment,
            backward_segments,
            is_first_rank,
            is_sp,
            is_amp_o1,
        ):
            """
            Transform one forward segment.
            """
            # 1. prepare the forward_segment
            #  1.1 check whether the forward_segment is right
            origin_matmul_op = global_block.ops[forward_segment[0]]
            origin_comm_op = global_block.ops[forward_segment[0] + 1]
            origin_add_op = global_block.ops[forward_segment[1] - 1]
            origin_cast_op = (
                global_block.ops[forward_segment[1] - 2] if is_amp_o1 else None
            )
            origin_matmul_output_name = origin_matmul_op.output_arg_names[0]
            origin_comm_input_name = origin_comm_op.input_arg_names[0]
            assert (
                origin_matmul_output_name == origin_comm_input_name
            ), f"The 0th op output name {origin_matmul_output_name} is not equal to the 1st op input name {origin_comm_input_name}"
            origin_comm_output_name = origin_comm_op.output_arg_names[0]
            origin_add_input_names = origin_add_op.input_arg_names
            assert (
                origin_comm_output_name == origin_add_input_names[0]
            ), f"The 1st op output name {origin_comm_output_name} is not equal to the 2nd op input name {origin_add_input_names[0]}"
            #  1.2 get the origin dist_attr
            origin_add_dist_attr = (
                self._dist_context.get_op_dist_attr_for_program(origin_add_op)
            )
            assert (
                origin_add_dist_attr is not None
            ), f"Origin add op {origin_add_op.type} has no dist attr"
            ref_mesh = origin_add_dist_attr.process_mesh
            in_var_dist_attr = origin_add_dist_attr.get_input_dist_attr(
                origin_add_op.input_arg_names[0]
            )
            ref_mapping = in_var_dist_attr.dims_mapping

            # 2. deal matmul_v2 op
            origin_matmul_output_new_name = unique_name.generate(
                origin_matmul_output_name + "@promote"
            )
            origin_matmul_output_new_var = global_block.create_var(
                name=origin_matmul_output_new_name,
                dtype=global_block.var(origin_matmul_output_name).dtype,
                shape=global_block.var(origin_matmul_output_name).shape,
                persistable=False,
                stop_gradient=False,
            )
            set_var_dist_attr(
                self._dist_context,
                origin_matmul_output_new_var,
                ref_mapping,
                ref_mesh,
            )
            rename_vars_map[origin_matmul_output_name] = (
                origin_matmul_output_new_name
            )
            origin_matmul_op._rename_output(
                origin_matmul_output_name, origin_matmul_output_new_name
            )
            naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
                origin_matmul_op, ref_mesh, ref_mapping, self._dist_context
            )

            # 3. deal add op and cast op
            if is_first_rank:
                # insert the "elementwise_add" op before reduce_sum
                new_add_op = global_block._insert_op_without_sync(
                    forward_segment[0] + 1,
                    type="nop",
                )
                new_op_desc = new_add_op.desc
                new_op_desc.copy_from(origin_add_op.desc)
                # create new var of new_add_op output
                origin_add_output_name = origin_add_op.output_arg_names[0]
                new_add_op_output_name = unique_name.generate(
                    origin_add_output_name + "@promote"
                )
                new_shape_var_name = (
                    origin_add_output_name
                    if not is_sp
                    else origin_matmul_output_name
                )
                global_block.create_var(
                    name=new_add_op_output_name,
                    dtype=global_block.var(origin_add_output_name).dtype,
                    shape=global_block.var(new_shape_var_name).shape,
                    persistable=False,
                    stop_gradient=False,
                )
                global_block._remove_var(
                    origin_matmul_output_name
                )  # We can remove the origin_matmul_output now.
                global_block._remove_var(origin_add_output_name)
                new_add_op._rename_output(
                    origin_add_output_name, new_add_op_output_name
                )
                rename_vars_map[origin_add_op.input_arg_names[0]] = (
                    origin_matmul_output_new_name
                )
                new_add_op._rename_input(
                    origin_add_op.input_arg_names[0],
                    origin_matmul_output_new_name,
                )
                # deal dist_attr
                naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
                    new_add_op, ref_mesh, ref_mapping, self._dist_context
                )
                # 'cast' op also need to adjust
                if is_amp_o1:
                    new_cast_op = global_block._insert_op_without_sync(
                        forward_segment[0] + 1,
                        type="nop",
                    )
                    new_op_desc = new_cast_op.desc
                    new_op_desc.copy_from(origin_cast_op.desc)
                    if (
                        new_cast_op.input_arg_names[0]
                        not in delete_bias_vars_name
                    ):  # fp16 = cast(fp32)
                        delete_bias_vars_name.append(
                            new_cast_op.input_arg_names[0]
                        )
                else:
                    if (
                        new_add_op.input_arg_names[1]
                        not in delete_bias_vars_name
                    ):
                        delete_bias_vars_name.append(
                            new_add_op.input_arg_names[1]
                        )
            else:
                # We can remove the origin_matmul_output now.
                origin_add_output_name = origin_add_op.output_arg_names[0]
                global_block._remove_var(origin_add_output_name)
                global_block._remove_var(origin_matmul_output_name)

            # 4. deal comm op
            # The input of c_allreduce_sum only be used once, so we don't need add it in the rename_vars_map
            if is_first_rank:
                origin_comm_op._rename_input(
                    origin_comm_op.input_arg_names[0],
                    new_add_op.output_arg_names[0],
                )
            else:
                origin_comm_op._rename_input(
                    origin_comm_op.input_arg_names[0],
                    origin_matmul_output_new_name,
                )
            if origin_comm_op.type == "c_allreduce_sum":
                new_comm_var_name = origin_comm_op.input_arg_names[0]
            else:
                new_comm_var_name = unique_name.generate(
                    origin_comm_output_name + "@promote"
                )
                global_block.create_var(
                    name=new_comm_var_name,
                    dtype=global_block.var(origin_comm_output_name).dtype,
                    shape=global_block.var(origin_comm_output_name).shape,
                    persistable=False,
                    stop_gradient=False,
                )
                rename_vars_map[origin_comm_output_name] = new_comm_var_name
            if global_block.has_var(origin_comm_output_name):
                global_block._remove_var(origin_comm_output_name)
            rename_vars_map[origin_add_output_name] = (
                new_comm_var_name  # the output of comm op inplace the output of add op for next ops
            )
            origin_comm_op._rename_output(
                origin_comm_output_name, new_comm_var_name
            )
            naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
                origin_comm_op, ref_mesh, ref_mapping, self._dist_context
            )

            # 5. remove elementwise_add op and cast op
            if is_first_rank:
                if is_amp_o1:
                    global_block._remove_op(forward_segment[0] + 5)
                    global_block._remove_op(forward_segment[0] + 4)
                else:
                    global_block._remove_op(forward_segment[0] + 3)
            else:
                global_block._remove_op(
                    forward_segment[1] - 1
                )  # remove elementwise_add op
                if is_amp_o1:
                    if (
                        origin_cast_op.input_arg_names[0]
                        not in delete_bias_vars_name
                    ):
                        delete_bias_vars_name.append(
                            origin_cast_op.input_arg_names[0]
                        )
                    global_block._remove_var(origin_cast_op.output_arg_names[0])
                    global_block._remove_op(
                        forward_segment[1] - 2
                    )  # remove cast op
                else:
                    if origin_add_input_names[1] not in delete_bias_vars_name:
                        delete_bias_vars_name.append(origin_add_input_names[1])
                # update backward forward_segment
                for back_seg in reversed(backward_segments):
                    if is_amp_o1:
                        if back_seg[0] > forward_segment[0]:
                            back_seg[0] -= 2
                            back_seg[1] -= 2
                        else:
                            break
                    else:
                        if back_seg[0] > forward_segment[0]:
                            back_seg[0] -= 1
                            back_seg[1] -= 1
                        else:
                            break

        global_block = main_program.global_block()
        rename_vars_map = {}  # origin_name -> new_name
        delete_bias_vars_name = []
        for segment in reversed(forward_segments):
            _transform_forward_segment(
                global_block,
                segment,
                backward_segments,
                is_first_rank,
                is_sp,
                is_amp_o1,
            )
        global_block._sync_with_cpp()
        return rename_vars_map, delete_bias_vars_name

    def _transform_backward(
        self,
        main_program,
        backward_segments,
        rename_var_names_map,
        is_first_rank,
        is_sp,
    ):
        global_block = main_program.global_block()
        to_delete_grad_of_param = []
        if is_first_rank:
            if is_sp:
                # place the comm_op(all_gather) before the elementwise_add_grad
                for segment in reversed(backward_segments):
                    add_grad_op = global_block.ops[segment[0]]
                    matmul_grad_op = global_block.ops[segment[-1] - 1]
                    origin_comm_op_id = segment[-1] - 2
                    origin_comm_op = global_block.ops[origin_comm_op_id]
                    new_comm_op = global_block._insert_op(
                        segment[0],
                        type="nop",
                    )
                    new_comm_op.desc.copy_from(origin_comm_op.desc)
                    # rename input and output
                    new_comm_op._rename_input(
                        origin_comm_op.input_arg_names[0],
                        add_grad_op.input_arg_names[0],
                    )
                    add_grad_op._rename_input(
                        add_grad_op.input_arg_names[0],
                        new_comm_op.output_arg_names[0],
                    )
                    matmul_grad_op._rename_input(
                        matmul_grad_op.input_arg_names[0],
                        add_grad_op.output_arg_names[0],
                    )

                    global_block._remove_op(segment[-1] - 1)
                    if self._enable_dp:
                        global_block._remove_op(segment[0] + 5)  # scale
                        global_block._remove_op(
                            segment[0] + 4
                        )  # c_allreduce_sum
                    else:
                        global_block._remove_op(segment[0] + 3)  # scale
                        global_block._remove_op(
                            segment[0] + 2
                        )  # c_allreduce_sum
                global_block._sync_with_cpp()
        else:  # not is_first_rank_in tp or sp
            # need to delete the grad op associated with the deleted bias var
            if not is_sp:
                for segment in reversed(backward_segments):
                    add_grad_op = global_block.ops[segment[0]]
                    rename_var_names_map[add_grad_op.output_arg_names[0]] = (
                        add_grad_op.input_arg_names[0]
                    )
                    global_block._remove_var(add_grad_op.output_arg_names[0])
                    to_delete_grad_of_param.append(
                        add_grad_op.output_arg_names[1]
                    )
                    if self._enable_dp:
                        global_block._remove_op(segment[0] + 2)  # scale op
                        global_block._remove_op(
                            segment[0] + 1
                        )  # c_allreduce_sum op
                    global_block._remove_op(segment[0])
                global_block._sync_with_cpp()
            else:
                for segment in reversed(backward_segments):
                    add_grad_op = global_block.ops[segment[0]]
                    origin_comm_op = global_block.ops[segment[-1] - 2]
                    rename_var_names_map[add_grad_op.output_arg_names[0]] = (
                        add_grad_op.input_arg_names[0]
                    )
                    origin_comm_op._rename_input(
                        origin_comm_op.input_arg_names[0],
                        add_grad_op.input_arg_names[0],
                    )
                    global_block._remove_var(add_grad_op.output_arg_names[0])
                    to_delete_grad_of_param.append(
                        add_grad_op.output_arg_names[1]
                    )
                    if self._enable_dp:  # DP
                        global_block._remove_op(
                            segment[0] + 4
                        )  # scale op for dp
                        global_block._remove_op(
                            segment[0] + 3
                        )  # c_allreduce_sum op for dp
                    global_block._remove_op(segment[0] + 2)  # scale op for sp
                    global_block._remove_op(
                        segment[0] + 1
                    )  # c_allreduce_sum op for sp
                    global_block._remove_op(
                        segment[0]
                    )  # elementwise_add_grad op
                global_block._sync_with_cpp()

        # rename input vars in global_block
        for op in global_block.ops:
            if is_optimize_op(op):
                continue
            for var_name in op.input_arg_names:
                if var_name in rename_var_names_map:
                    op._rename_input(var_name, rename_var_names_map[var_name])
        if self._is_amp_o1:
            for var_name in to_delete_grad_of_param:
                global_block._remove_var(var_name)
        global_block._sync_with_cpp()

    def _transform_opt(
        self,
        main_program,
        deleted_bias_names,
        params_grads,
        is_first_rank,
        is_amp_o1,
    ):
        if is_first_rank:
            return
        deleted_bias_grads_names = []
        to_delete_params_grads = []
        for id, (param, grad) in enumerate(params_grads):
            if param.name in deleted_bias_names:
                deleted_bias_grads_names.append(grad.name)
                to_delete_params_grads.append(id)

        to_delete_op_ids = []
        for id in reversed(range(len(main_program.global_block().ops))):
            global_block = main_program.global_block()
            op = global_block.ops[id]
            op_input_names = op.input_arg_names
            for op_input in op_input_names:
                if op_input in deleted_bias_grads_names:
                    if op.type in _supported_optimizer_type:
                        for output_var in op.output_arg_names:
                            global_block._remove_var(output_var)
                        grad_var = op.input('Grad')[0]
                        global_block._remove_var(grad_var)
                        to_delete_op_ids.append(id)
                    if (
                        op.type == "squared_l2_norm"
                        or op.type == "clip_by_norm"
                    ):
                        output_var_name = op.output_arg_names[0]
                        global_block._remove_var(output_var_name)
                        to_delete_op_ids.append(id)
                        for intra_id in range(id + 1, len(global_block.ops)):
                            intra_op = global_block.ops[intra_id]
                            if (
                                output_var_name in intra_op.input_arg_names
                                and intra_op.type == "stack"
                            ):
                                origin_vars = intra_op.input("X")
                                origin_vars.remove(output_var_name)
                                intra_op.desc.set_input("X", origin_vars)
                                break
                    if op.type == "elementwise_mul":
                        to_delete_op_ids.append(id)
                    # check_finite_and_unscale and update_loss_scaling
                    if (
                        op.type == "check_finite_and_unscale"
                        or op.type == "update_loss_scaling"
                    ):
                        origin_vars = op.input("X")
                        origin_vars.remove(op_input)
                        op.desc.set_input("X", origin_vars)
                        origin_vars = op.output("Out")
                        origin_vars.remove(op_input)
                        op.desc.set_output("Out", origin_vars)

            if is_amp_o1:
                for output_name in op.output_arg_names:
                    if (
                        output_name in deleted_bias_grads_names
                        and op.type == 'cast'
                    ):
                        to_delete_op_ids.append(id)

        for id in to_delete_op_ids:
            global_block._remove_op(id)
        main_program.global_block()._sync_with_cpp()

        for id in reversed(to_delete_params_grads):
            del params_grads[id]
        return

    def _transform_startup_program(
        self, startup_program, deleted_bias_names, dp_group, is_first_rank
    ):
        """
        Delete the vars and ops associated with deleted_bias_names in startup program.
        """
        logger.debug(f"Before transform startup_program: {startup_program}")
        cur_glock = startup_program.global_block()
        to_delete_op_ids = []
        # for variables associated with deleted_bias_names in amp-o2, such as 'opt_linear_1.b_0_fp32_master_0'
        to_delete_extra_vars = []
        for id, op in enumerate(cur_glock.ops):
            if not is_first_rank:
                output_var = op.output_arg_names[0]
                if output_var in deleted_bias_names:
                    to_delete_op_ids.append(id)
                else:
                    for var_name in deleted_bias_names:
                        if var_name in output_var:
                            to_delete_op_ids.append(id)
                            if output_var not in to_delete_extra_vars:
                                to_delete_extra_vars.append(output_var)
            else:
                if op.type == "broadcast":
                    input_vars = op.input_arg_names
                    if (
                        input_vars[0] in deleted_bias_names
                        and id not in to_delete_op_ids
                    ):
                        if dp_group is None or (
                            dp_group is not None
                            and op.attr("ring_id") != dp_group.id
                        ):
                            to_delete_op_ids.append(id)
        for to_delete_id in reversed(to_delete_op_ids):
            cur_glock._remove_op(to_delete_id)
        if not is_first_rank:
            for var_name in deleted_bias_names:
                cur_glock._remove_var(var_name)
            for var_name in to_delete_extra_vars:
                if cur_glock.has_var(var_name):
                    cur_glock._remove_var(var_name)
        cur_glock._sync_with_cpp()
        logger.debug(f"After transform startup_program: {startup_program}")
