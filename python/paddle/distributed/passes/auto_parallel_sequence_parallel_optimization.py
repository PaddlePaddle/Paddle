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

import paddle
from paddle.distributed.auto_parallel.utils import _get_comm_group
from paddle.framework import core
from paddle.utils import unique_name

OpRole = core.op_proto_and_checker_maker.OpRole
OP_ROLE_KEY = core.op_proto_and_checker_maker.kOpRoleAttrName()
OP_ROLE_VAR_KEY = core.op_proto_and_checker_maker.kOpRoleVarAttrName()

from .pass_base import PassBase, PassType, register_pass


@register_pass("auto_parallel_sequence_parallel_optimization")
class SequenceParallelOptimization(PassBase):
    """
    Currently, the pass can only detect this pattern:
    1. with hidden dropout
    2. with residual add
    3. with pre layer norm
    4. full granularity when use recompute

    These following config won't affect the pass:
    1. way to compute qkv
    2. attn dropout or not
    3. use recompute or not
    4. use gradient merge or not

    To support all transformer pattern, here are the following jobs:
    1. add pass config to control the pattern detector
    2. add pass config to change some assertions in op replacement

    The op replacement should be same regardless of the transformer structure.
    """

    # TODO(Yuang Liu): make the pass more general: support no dropout version

    def __init__(self):
        super().__init__()
        self.world_size = paddle.distributed.get_world_size()

        # Record the origin var name to new var name, which are created for forward and recompute.
        # Such as c_allgather, c_reducescatter output var.
        self._name_mapping = {}

        # Record the idx where we insert extra op
        self._extra_op_idx = []

        # Record grad that need an extra allreudce sum.
        # Layer norm bias and weight. Row parallel bias.
        self._grad_need_allreduce = []

        self._ring_id = None

    def _check_self(self):
        if self.get_attr("dist_context") is None:
            return False
        if self.get_attr("global_rank") is None:
            return False
        return True

    def _check_conflict(self, other_pass):
        # add conflict checker if meet conflict in the future
        return True

    def _type(self):
        return PassType.CALC_OPT

    def _get_pp_stage(self, rank, dist_context):
        pp_idx = None
        for idx, process_mesh in enumerate(dist_context.process_meshes):
            if rank in process_mesh.process_ids:
                pp_idx = idx
                break
        return pp_idx

    def _parse_hybrid_degree(self, program):
        dist_context = self._attrs["dist_context"]
        main_block = program.block(0)
        mp_degree = -1
        mp_group_ranks = None
        for param in main_block.all_parameters():
            param_context = dist_context.get_tensor_dist_attr_for_program(param)
            process_mesh = param_context.process_mesh
            process_mesh_shape = process_mesh.shape
            process_mesh_size = len(process_mesh.process_ids)
            assert (
                self.world_size % process_mesh_size == 0
            ), "The world size has to be divided by the size of process mesh"
            dims_mapping = param_context.dims_mapping
            for value in dims_mapping:
                if value != -1:
                    # Here, we assume that only mp will split parameters. So, when the dims mapping
                    # of a parameter is not -1, it means the parameter has been split due to mp.
                    if mp_degree == -1:
                        mp_degree = process_mesh_shape[value]
                    else:
                        # Here is another assumption: all parameters' mp degree
                        # are the same under SEMI AUTO scenario.
                        assert mp_degree == process_mesh_shape[value]

            # get mp stage
            mp_size_axis = -1
            for mapping in dims_mapping:
                # get the dims, which that be split by mp
                if mapping != -1:
                    assert (
                        mp_size_axis == -1
                    ), "param only can be split by one mesh"
                    mp_size_axis = mapping
            if mp_size_axis == -1:
                continue
            # get the group that mp used to comm between each ranks
            group_ranks = _get_comm_group(
                process_mesh.process_ids,
                process_mesh.shape,
                mp_size_axis,
                self.local_rank,
            )
            if mp_group_ranks is None:
                mp_group_ranks = group_ranks
            else:
                assert mp_group_ranks == group_ranks

        self.mp_degree = mp_degree
        self.pp_degree = len(dist_context.process_meshes)
        assert mp_group_ranks is not None
        self.mp_stage = mp_group_ranks.index(self.local_rank)
        self.pp_stage = self._get_pp_stage(self.local_rank, dist_context)

        assert (
            self.world_size % (self.mp_degree * self.pp_degree) == 0
        ), "The world size has to be divided by pp_degree * mp_degree"
        self.dp_degree = int(
            self.world_size / (self.mp_degree * self.pp_degree)
        )

        self.is_pp_first_stage = self.pp_stage == 0
        self.is_pp_last_stage = self.pp_stage == (self.pp_degree - 1)
        logging.info(
            f"parsed mp_degree: {self.mp_degree}, pp_degree: {self.pp_degree}, "
            f"pp_first_stage: {self.is_pp_first_stage}, pp_last_stage: {self.is_pp_last_stage}, "
            f"mp_stage: {self.mp_stage}"
        )

    def _get_hyper_params(self, main_prog, ref_op_idx):
        # parse batch size, sequence length and hidden size from the layer norm op
        block = main_prog.block(0)
        ops = list(block.ops)[ref_op_idx]
        x = block.var(ops.input("X")[0])
        shape = x.shape
        assert len(shape) == 3
        self.batch_size = shape[0]
        self.seq_len = shape[1]
        self.hidden_size = shape[2]
        assert self.seq_len % self.mp_degree == 0

    def _create_var_helper(self, block, name, ref_var, shape=None):
        var = block.create_var(
            name=name,
            shape=ref_var.shape if shape is None else shape,
            dtype=ref_var.dtype,
            type=ref_var.type,
            lod_level=ref_var.lod_level,
            persistable=ref_var.persistable,
            is_data=ref_var.is_data,
            need_check_feed=ref_var.desc.need_check_feed(),
        )
        return var

    def _new_var_name_replace(self, origin_name, new_var, idx, end_idx, ops):
        # Use the output of new op's (c_allgather or c_reducescatter) output
        # to replace all input using origin op's (c_identity or c_allreduce_sum) output.
        for scan_idx in range(idx + 1, end_idx + 1):
            scanning_op = ops[scan_idx]
            for input_name in scanning_op.input_names:
                if origin_name in scanning_op.input(input_name):
                    all_input = scanning_op.input(input_name)
                    idx = all_input.index(origin_name)
                    all_input.remove(origin_name)
                    all_input.insert(idx, new_var.name)
                    scanning_op.desc.set_input(input_name, all_input)

    def _ring_id_getter_or_checker(self, ops, start_idx, end_idx):
        for idx in range(end_idx, start_idx - 1, -1):
            op = ops[idx]
            if op.type == 'c_allreduce_sum' or op.type == 'c_identity':
                if op.has_attr("use_model_parallel") and not op.attr(
                    "use_model_parallel"
                ):
                    # skip the all reduce not for mp
                    continue
                if self._ring_id is None:
                    self._ring_id = op.attr("ring_id")
                else:
                    assert self._ring_id == op.attr(
                        "ring_id"
                    ), "mp should have same ring id"

    def _c_identity_op_replace(
        self,
        block,
        op,
        idx,
        op_role,
        end_idx,
        forward=True,
        recompute=False,
    ):
        # Replace c_identity with c_allgather.
        # SP has split the sequence dim for layer norm and dropout etc.
        # Here we re-combine the sequence dim for the coming core attn or ffn.
        suffix = "_forward"
        if recompute:
            suffix = "_recompute"
        elif not forward:
            suffix = "_backward"
        x_var = block.var(op.input("X")[0])
        y_name = op.output("Out")[0]
        assert self._ring_id is not None
        block._remove_op(idx, sync=False)
        allgather_out_var = self._create_var_helper(
            block,
            unique_name.generate("c_allgather_output_var" + suffix),
            x_var,
            [self.seq_len, self.batch_size, self.hidden_size],
        )
        block._insert_op_without_sync(
            idx,
            type='c_allgather',
            inputs={'X': [x_var]},
            outputs={'Out': [allgather_out_var]},
            attrs={
                'ring_id': self._ring_id,
                'use_calc_stream': True,
                'nranks': self.mp_degree,
                OP_ROLE_KEY: op_role,
            },
        )
        if forward or recompute:
            self._name_mapping[x_var.name] = allgather_out_var.name
            self._name_mapping[y_name] = allgather_out_var.name
        self._new_var_name_replace(
            y_name, allgather_out_var, idx, end_idx, list(block.ops)
        )

    def _c_allreduce_sum_op_replace(
        self,
        block,
        op,
        idx,
        op_role,
        end_idx,
        forward=True,
        recompute=False,
    ):
        suffix = "_forward"
        if recompute:
            suffix = "_recompute"
        elif not forward:
            suffix = "_backward"
        if op.has_attr("use_model_parallel") and not op.attr(
            "use_model_parallel"
        ):
            # skip the all reduce not for mp
            return
        # Replace the c_allreduce_sum with reduce_scatter.
        # After the core attn and ffn, we should split the sequence dim again.
        x_var = block.var(op.input("X")[0])
        y_name = op.output("Out")[0]
        assert self._ring_id is not None
        block._remove_op(idx, sync=False)
        reduce_scatter_out_var = self._create_var_helper(
            block,
            unique_name.generate("reduce_scatter_output_var" + suffix),
            x_var,
            [
                self.seq_len // self.mp_degree,
                self.batch_size,
                self.hidden_size,
            ],
        )
        if x_var.shape[0] % self.mp_degree != 0:
            # When inserting op, the x shape is not correct, just hack change the shape to pass
            # infer shape for the op insertion. The correct shape will be updated later this pass.
            x_var_shape = list(x_var.shape)
            x_var_shape[0] = self.mp_degree
            x_var.desc.set_shape(x_var_shape)
        block._insert_op_without_sync(
            idx,
            type='c_reducescatter',
            inputs={'X': [x_var]},
            outputs={'Out': [reduce_scatter_out_var]},
            attrs={
                'ring_id': self._ring_id,
                'use_calc_stream': True,
                'nranks': self.mp_degree,
                OP_ROLE_KEY: op_role,
            },
        )
        if forward or recompute:
            self._name_mapping[y_name] = reduce_scatter_out_var.name
        self._new_var_name_replace(
            y_name, reduce_scatter_out_var, idx, end_idx, list(block.ops)
        )

    def _transpose_perm_replacement(
        self, appear_transpose_count, op, forward=True
    ):
        # Replace the perm of transpose 2.
        # Before the pass, the perm is [0, 2, 1, 3],
        # which is [b, s, n_head, head_dim] -> [b, n_head, s, head_dim], (before core attn (w/wo grad))
        # or [b, n_head, s, head_dim] -> [b, s, n_head, head_dim]. (after core attn (w/wo grad))
        # After the pass, the perm before core attn (w/wo grad) is [1, 2, 0, 3],
        # which is [s, b, n_head, head_dim] -> [b, n_head, s, head_dim],
        # and the perm after core attn (w/wo grad) is [2, 0, 1, 3],
        # which is [b, n_head, s, head_dim] -> [s, b, n_head, head_dim].
        assert op.has_attr('axis')
        origin_perm = op.attr('axis')
        assert origin_perm == [0, 2, 1, 3]
        if (forward and appear_transpose_count == 0) or (
            not forward
            and appear_transpose_count != self.c_identity_num_for_qkv
        ):
            # Travers backward, the first transpose is post core attn or before core attn grad.
            op._set_attr('axis', [2, 0, 1, 3])
        else:
            # All reset transpose (maybe 1, 2 or 3) are all pre core attn or after core attn grad.
            op._set_attr('axis', [1, 2, 0, 3])
        appear_transpose_count += 1
        return appear_transpose_count

    def _input_data_or_grad_handler(
        self,
        block,
        op,
        idx,
        op_role,
        end_idx,
        forward=True,
        recompute=False,
    ):
        # For now, the pass only support pre layer norm pattern.
        # Remove this assertion later to support more transformer pattern.
        if forward:
            assert op.type == 'layer_norm'
        else:
            assert op.type == 'elementwise_add_grad'

        suffix = "_forward"
        if recompute:
            suffix = "_recompute"
        elif not forward:
            suffix = "_backward"
        prefix = "sp_first_block_" if forward else "sp_last_block_"
        handle_input_name = "X" if forward else "Out@GRAD"

        anchor_var = block.var(op.input(handle_input_name)[0])
        # Insert a transpose op to change the input from [b, s, h] to [s, b, h].
        # Insert a split op to split the sequence dim.
        x_shape_var = self._create_var_helper(
            block,
            unique_name.generate(prefix + "transpose_x_shape" + suffix),
            anchor_var,
        )
        transpose_out_var = self._create_var_helper(
            block,
            unique_name.generate(prefix + "transpose_out" + suffix),
            anchor_var,
            [self.seq_len, self.batch_size, self.hidden_size],
        )
        block._insert_op_without_sync(
            idx,
            type='transpose2',
            inputs={'X': [anchor_var]},
            outputs={
                'Out': [transpose_out_var],
                'XShape': [x_shape_var],
            },
            attrs={'axis': [1, 0, 2], OP_ROLE_KEY: op_role},
        )
        split_out_vars = []
        for i in range(self.mp_degree):
            split_out_vars.append(
                self._create_var_helper(
                    block,
                    unique_name.generate(prefix + f"split_out_{i}" + suffix),
                    anchor_var,
                    [
                        self.seq_len // self.mp_degree,
                        self.batch_size,
                        self.hidden_size,
                    ],
                )
            )
        block._insert_op_without_sync(
            idx + 1,
            type='split',
            inputs={'X': [transpose_out_var]},
            outputs={'Out': split_out_vars},
            attrs={
                'axis': 0,
                'num': self.mp_degree,
                OP_ROLE_KEY: op_role,
            },
        )
        new_op_input_var = split_out_vars[self.mp_stage]
        op.desc.set_input(handle_input_name, [new_op_input_var.name])
        if forward or recompute:
            self._name_mapping[anchor_var.name] = new_op_input_var.name
            self._new_var_name_replace(
                anchor_var.name, new_op_input_var, idx, end_idx, list(block.ops)
            )
        self._extra_op_idx.append(idx)

    def _output_data_or_grad_handler(
        self, block, op, ops, idx, op_role, forward=True
    ):
        if forward:
            # For now, the pass only support pre layer norm pattern.
            # Remove this assertion later to support more transformer pattern.
            assert op.type == "elementwise_add"
            next_op = ops[idx + 1]
            # The op after the last transformer block should be a layer norm for the encoder layer.
            assert next_op.type == "layer_norm"
        else:
            assert op.type == 'layer_norm_grad'
            # During backward pattern detection, we skip the sum op for residual add.
            # But when handle the backward for the first transformer block,
            # we have to handle the sum op. To make sure the grad pass to the data is correct.
            while True:
                idx = idx + 1
                sum_op = ops[idx]
                assert (
                    sum_op.type == 'sum' or sum_op.type == 'c_allreduce_sum'
                ), 'only support residual add scenario'
                if sum_op.type == 'sum':
                    break
            op = sum_op
            next_op = ops[idx + 1]
            assert (
                next_op.type == 'dropout_grad'
            ), "The sequence parallel pass only support transformer structure with hidden dropout."

        suffix = "_forward" if forward else "_backward"
        prefix = "sp_last_block_" if forward else "sp_first_block_"

        all_gather_input_var = block.var(op.output("Out")[0])
        # Insert an allgather op to gather all sub tensor on each device to get full output.
        # Insert a transpose op to change the final output from [s, b, h] to [b, s, h].
        allgather_out_var = self._create_var_helper(
            block,
            unique_name.generate(prefix + "allgather_out" + suffix),
            all_gather_input_var,
            [self.seq_len, self.batch_size, self.hidden_size],
        )
        block._insert_op_without_sync(
            idx + 1,
            type='c_allgather',
            inputs={'X': [all_gather_input_var]},
            outputs={'Out': [allgather_out_var]},
            attrs={
                'ring_id': self._ring_id,
                'use_calc_stream': True,
                'nranks': self.mp_degree,
                OP_ROLE_KEY: op_role,
            },
        )
        x_shape_var = self._create_var_helper(
            block,
            unique_name.generate(prefix + "transpose_x_shape" + suffix),
            all_gather_input_var,
        )
        transpose_out_var = self._create_var_helper(
            block,
            unique_name.generate(prefix + "transpose_out" + suffix),
            all_gather_input_var,
            [self.batch_size, self.seq_len, self.hidden_size],
        )
        block._insert_op_without_sync(
            idx + 2,
            type='transpose2',
            inputs={'X': [allgather_out_var]},
            outputs={
                'Out': [transpose_out_var],
                'XShape': [x_shape_var],
            },
            attrs={'axis': [1, 0, 2], OP_ROLE_KEY: op_role},
        )
        # Use the updated output to replace the origin output.
        anchor_name = "X" if forward else "Out@GRAD"
        next_op_input = next_op.input(anchor_name)[0]
        if forward:
            self._name_mapping[
                all_gather_input_var.name
            ] = transpose_out_var.name
        assert next_op_input == all_gather_input_var.name
        next_op.desc.set_input(anchor_name, [transpose_out_var.name])
        self._extra_op_idx.append(idx + 1)

    def _forward_transformer_block_replace(
        self,
        main_prog,
        start_idx,
        end_idx,
        first=False,
        last=False,
        for_recompute=False,
    ):
        op_role = OpRole.Forward if not for_recompute else OpRole.Backward
        block = main_prog.block(0)
        # Pre loop all ops in block, to determine ring id.
        # There will be two kind of transpose, which has different updated perm.
        # We traverse op backward, the last op need ring id only can be got in previous op.
        ops = list(block.ops)
        self._ring_id_getter_or_checker(ops, start_idx, end_idx)

        appear_transpose_count = 0
        for idx in range(end_idx, start_idx - 1, -1):
            # reverse replace the op
            ops = list(block.ops)
            op = ops[idx]
            if idx == start_idx and first:
                self._input_data_or_grad_handler(
                    block,
                    op,
                    idx,
                    op_role,
                    end_idx,
                    forward=True,
                    recompute=for_recompute,
                )
            elif op.type == 'c_identity':
                self._c_identity_op_replace(
                    block,
                    op,
                    idx,
                    op_role,
                    end_idx,
                    forward=True,
                    recompute=for_recompute,
                )
            elif op.type == 'transpose2':
                appear_transpose_count = self._transpose_perm_replacement(
                    appear_transpose_count, op
                )
            elif op.type == 'c_allreduce_sum':
                self._c_allreduce_sum_op_replace(
                    block,
                    op,
                    idx,
                    op_role,
                    end_idx,
                    forward=True,
                    recompute=for_recompute,
                )
            elif idx == end_idx and last and not for_recompute:
                self._output_data_or_grad_handler(
                    block, op, ops, idx, op_role, forward=True
                )
        block._sync_with_cpp()
        return main_prog

    def _backward_input_replacing(self, main_prog):
        # Update the recompute section idx according to the inserted op
        for idx in range(len(self._recompute_locations)):
            for new_op_idx in self._extra_op_idx:
                if new_op_idx < self._recompute_locations[idx][0]:
                    # The inserted op is before the recompute section,
                    # each new op idx will insert two new ops:
                    # transpose + split or transpose + allgather.
                    self._recompute_locations[idx][0] += 2
                    self._recompute_locations[idx][1] += 2
                elif new_op_idx == self._recompute_locations[idx][0] or (
                    self._recompute_locations[idx][0]
                    < new_op_idx
                    <= self._recompute_locations[idx][1]
                ):
                    # The inserted op is inside the recompute section,
                    # only need to update the end of the section
                    self._recompute_locations[idx][1] += 2

        block = main_prog.block(0)
        ops = list(block.ops)
        for idx, op in enumerate(ops):
            # Use the updated X for the backward computation
            if (
                'X' in op.input_names
                and self._name_mapping.get(op.input('X')[0], None) is not None
                and op.attr('op_role') == int(OpRole.Backward)
            ):
                inner_recompute = False
                for recompute_location in self._recompute_locations:
                    # Skip the op for recompute
                    if recompute_location[0] <= idx <= recompute_location[1]:
                        inner_recompute = True
                if inner_recompute:
                    continue
                new_name = self._name_mapping.get(op.input('X')[0])
                op.desc.set_input("X", [new_name])
        block._sync_with_cpp()
        return main_prog

    def _backward_transformer_block_replace(
        self, main_prog, start_idx, end_idx, first=False, last=False
    ):
        op_role = OpRole.Backward
        block = main_prog.block(0)
        # Pre loop all ops in block, to determine ring id.
        # There will be two kind of transpose, which has different updated perm.
        # We traverse op backward, the last op need ring id only can be got in previous op.
        ops = list(block.ops)
        self._ring_id_getter_or_checker(ops, start_idx, end_idx)

        appear_transpose_count = 0
        for idx in range(end_idx, start_idx - 1, -1):
            # reverse replace the op
            ops = list(block.ops)
            op = ops[idx]

            # The bias and weight of layer norm and the bias of column parallel linear
            # only contains partial grad for sequence parallel (it lost the info which
            # contained by other sp device). Some extra allreduce sum operations are
            # needed to make sure these parameters are updated correctly.
            if op.type == 'layer_norm_grad':
                self._grad_need_allreduce.append(op.output('Bias@GRAD')[0])
                self._grad_need_allreduce.append(op.output('Scale@GRAD')[0])
            if (
                op.type == 'c_identity'
                and ops[idx - 1].type == 'elementwise_add_grad'
            ):
                self._grad_need_allreduce.append(
                    ops[idx - 1].output('Y@GRAD')[0]
                )

            if idx == start_idx and last:
                self._input_data_or_grad_handler(
                    block,
                    op,
                    idx,
                    op_role,
                    end_idx,
                    forward=False,
                    recompute=False,
                )
            elif op.type == 'c_identity':
                self._c_identity_op_replace(
                    block,
                    op,
                    idx,
                    op_role,
                    end_idx,
                    forward=False,
                    recompute=False,
                )
            elif op.type == 'transpose2_grad':
                appear_transpose_count = self._transpose_perm_replacement(
                    appear_transpose_count, op, False
                )
            elif op.type == 'c_allreduce_sum':
                self._c_allreduce_sum_op_replace(
                    block,
                    op,
                    idx,
                    op_role,
                    end_idx,
                    forward=False,
                    recompute=False,
                )
            elif idx == end_idx and first:
                self._output_data_or_grad_handler(
                    block, op, ops, idx, op_role, forward=False
                )
        block._sync_with_cpp()
        return main_prog

    def _infer_shape_updadte(self, main_prog):
        block = main_prog.block(0)
        ops = list(block.ops)
        if not self.is_pp_first_stage:
            # change the shape of recv v2 var, for following infer shape
            recv_op = ops[0]
            assert recv_op.type == 'recv_v2'
            var = block.var(recv_op.output('Out')[0])
            var.desc.set_shape(
                [
                    self.seq_len // self.mp_degree,
                    self.batch_size,
                    self.hidden_size,
                ]
            )
        for op in ops:
            op.desc.infer_shape(block.desc)
        block._sync_with_cpp()
        if self._gradient_merge:
            # Update the gradient merge block's ops
            sub_block = None
            for idx, op in enumerate(reversed(ops)):
                # Travers all op backwardly, meet conditional block more quick.
                if op.type == 'conditional_block':
                    assert op.has_attr('sub_block')
                    sub_block = op.attr('sub_block')
                    break
            assert sub_block is not None
            block = main_prog.block(sub_block.id)
            ops = list(block.ops)
            for op in ops:
                op.desc.infer_shape(block.desc)
            block._sync_with_cpp()
        return main_prog

    def _insert_allreduce(self, main_prog):
        block = main_prog.block(0)
        ops = list(block.ops)
        insert_idx = -1
        for idx, op in enumerate(ops):
            # parse the grad to the main grad which is used to update params
            if op.type == 'cast':
                x_name = op.input('X')[0]
                if x_name not in self._grad_need_allreduce:
                    continue
                index = self._grad_need_allreduce.index(x_name)
                self._grad_need_allreduce.remove(x_name)
                self._grad_need_allreduce.insert(index, op.output('Out')[0])
            if op.attr('op_role') == int(OpRole.Optimize) and insert_idx == -1:
                insert_idx = idx

        if self._gradient_merge:
            # Find the correct block to insert extra c allreduce
            sub_block = None
            for idx, op in enumerate(reversed(ops)):
                # Travers all op backwardly, meet conditional block more quick.
                if op.type == 'conditional_block':
                    assert op.has_attr('sub_block')
                    sub_block = op.attr('sub_block')
                    break
            assert (
                sub_block is not None
            ), "Gradient merge should have sub block."
            grad_merge_vars = []
            for grad in self._grad_need_allreduce:
                grad_merge_var_name = grad + '@GradientMerge'
                assert block.has_var(grad_merge_var_name)
                grad_merge_vars.append(grad_merge_var_name)
            self._grad_need_allreduce = grad_merge_vars
            block = main_prog.block(sub_block.id)
            insert_idx = 0

        for grad in self._grad_need_allreduce:
            # For gradient merge, the var cannot be found in the gradient merge block.
            # Have to create op desc from block desc, then insert the op back to the grad merge block.
            allreduce_op_desc = block.desc._insert_op(insert_idx)
            allreduce_op_desc.set_type('c_allreduce_sum')
            allreduce_op_desc.set_input('X', [grad])
            allreduce_op_desc.set_output('Out', [grad])
            allreduce_op_desc._set_attr('ring_id', self._ring_id)
            allreduce_op_desc._set_attr('use_calc_stream', True)
            allreduce_op_desc._set_attr('op_role', OpRole.Backward)

            # Sync the python block with cpp block
            allreduce_op = paddle.static.Operator(block, allreduce_op_desc)
            block.ops.insert(insert_idx, allreduce_op)

        block._sync_with_cpp()
        return main_prog

    def _locate_transformer_block(self, program):
        # For forward:
        # 1. locate the first c_identity for mp (op before the all reduce is forward)
        # 2. from the location, search backward to the nearest layer norm (pre ln)
        # 3. locate the second c_identity for mp
        # 4. from the new location, search forward to the third element wise add (out linear bias and residual)

        # For backward:
        # 1. location the first c_identity for mp (op before the all reduce is backward)
        # 2. from the location, search backward to the second element wise add grad (two ffn and one residual)
        # 3. locate the second c_identity for mp
        # 4. from the new location, search forward to the nearest layer_norm_grad op
        #    (ignore the sum op for residual, only handle it for the first transformer block)

        # For recompute: mostly same with for forward, but only detect the nearest dropout op,
        # since recompute block won't recompute the checkpoint.

        # return: a dict,
        # The key is forward or backward.
        # The value of each key is a list of pair of start idx and end idx for a transformer block.
        # For example:
        # {
        #   "forward": [[2, 30], [31, 59]],
        #   "backward": [[90, 112], [124, 156]],
        #   "recompute": [[60, 89], [113, 123]]
        # }
        # Explanation: there are two forward transformer blocks and two backward transformer blocks.
        # The location of the two forward blocks are from 2 to 30 and from 32 to 59.
        # The location of the two backward blocks are from 90 to 122 and from 124 to 156.
        # The location of the two recompute blocks are from 60 to 89 and from 113 to 123.
        # Note: the idx is the position for the related op in the main program.
        ops = list(program.block(0).ops)

        # Pre loop part of the ops to determine the calc method of qkv. (Fuse qkv, fuse kv or etc.)
        self.c_identity_num_for_qkv = 0
        for idx, op in enumerate(ops):
            if (
                op.type == 'c_identity'
                and op.has_attr("use_model_parallel")
                and op.attr("use_model_parallel")
            ):
                self.c_identity_num_for_qkv += 1
            if (
                op.type == 'c_allreduce_sum'
                and op.has_attr("use_model_parallel")
                and op.attr("use_model_parallel")
                and self.c_identity_num_for_qkv > 0
            ):
                break

        parsing_backward = False
        c_identity_num = 0
        detect_recompute = False
        forward_locations = []
        backward_locations = []
        recompute_locations = []
        start_idx = -1
        end_idx = -1
        for idx, op in enumerate(ops):
            if not parsing_backward and (
                'grad' in op.type or op.type == 'send_v2'
            ):
                # Reset same flag values when first meet grad op.
                # There may be one extra c_identity for embedding weight sharing in the forward program.
                # Two ways to determine the start of backward:
                # 1. meet grad op (for last pp stage or non pp mode).
                # 2. meet recv_v2 op (for no last pp stage under pp mode).
                start_idx = -1
                end_idx = -1
                c_identity_num = 0
                parsing_backward = True
                detect_recompute = True
            if (
                op.type == 'c_identity'
                and op.has_attr("use_model_parallel")
                and op.attr("use_model_parallel")
            ):
                if (not parsing_backward) or (
                    detect_recompute and self._recompute
                ):
                    # not parsing_backward: locate the forward transformer block
                    # detect_recompute and self._recompute: locate recompute transformer block
                    if c_identity_num == 0:
                        c_identity_num += 1
                        for i in range(idx, 0, -1):
                            # search backward for the layer norm
                            if ops[i].type == 'layer_norm':
                                start_idx = i
                                break
                    else:
                        c_identity_num += 1
                        # For forward, except c identity for qkv, there will be one more c identity for ffn.
                        if c_identity_num < self.c_identity_num_for_qkv + 1:
                            continue
                        c_identity_num = 0
                        elementwise_add_count = 0
                        for i in range(idx, len(ops)):
                            # search forward for two element wise add
                            if (
                                not parsing_backward
                                and ops[i].type == 'elementwise_add'
                            ):
                                elementwise_add_count += 1
                                if elementwise_add_count == 3:
                                    end_idx = i
                                    break
                            else:
                                if (
                                    detect_recompute and self._recompute
                                ) and ops[i].type == 'dropout':
                                    detect_recompute = False
                                    end_idx = i
                                    break
                        if not parsing_backward:
                            forward_locations.append([start_idx, end_idx])
                        else:
                            recompute_locations.append([start_idx, end_idx])
                        start_idx = -1
                        end_idx = -1
                else:
                    if c_identity_num == 0:
                        elementwise_add_grad_count = 0
                        for i in range(idx, 0, -1):
                            # search backward for three elementwise add grad
                            if ops[i].type == 'elementwise_add_grad':
                                elementwise_add_grad_count += 1
                                if elementwise_add_grad_count == 2:
                                    start_idx = i
                                    break
                        # for backward, only one c identity no matter how qkv are computed
                        c_identity_num = 1
                    else:
                        c_identity_num = 0
                        for i in range(idx, len(ops)):
                            # search forward to the sum
                            if ops[i].type == 'layer_norm_grad':
                                end_idx = i
                                break
                        backward_locations.append([start_idx, end_idx])
                        start_idx = -1
                        end_idx = -1
                        # recompute block and backward block will show alternately
                        detect_recompute = True

        return {
            "forward": forward_locations,
            "backward": backward_locations,
            "recompute": recompute_locations,
        }

    def _location_validation_check(
        self, forward_positions, backward_positions, recompute_locations
    ):
        if not self._recompute:
            assert (
                len(recompute_locations) == 0
            ), "detect recompute pattern while no recompute"

        # Validate the position parsed by the pass.
        # This is a redundant checker to ensure everything goes on the right track.
        # Can be removed for performance concern.
        # The forward positions and backward positions are already sorted by the start idx.
        if forward_positions[0][1] >= backward_positions[-1][0]:
            # forward ops overlap with backward ops
            return False
        if (
            self._recompute
            and forward_positions[0][1] >= recompute_locations[-1][0]
        ):
            # forward ops overlap with recompute ops
            return False
        idx_set = [-1]
        for pos in reversed(forward_positions):
            if pos[0] >= pos[1]:
                # current pos have to be increasing
                return False
            if idx_set.count(pos[0]) > 0 or idx_set.count(pos[1]) > 0:
                # the boundary should be unique
                return False
            if pos[0] <= max(idx_set):
                # the positions should be increasing
                return False
            idx_set.append(pos[0])
            idx_set.append(pos[1])
        for pos in reversed(backward_positions):
            if pos[0] >= pos[1]:
                # current pos have to be increasing
                return False
            if idx_set.count(pos[0]) > 0 or idx_set.count(pos[1]) > 0:
                # the boundary should be unique
                return False
            if pos[0] <= max(idx_set):
                # the positions should be increasing
                return False
            idx_set.append(pos[0])
            idx_set.append(pos[1])
        if self._recompute:
            for pos in reversed(recompute_locations):
                if pos[0] >= pos[1]:
                    # current pos have to be increasing
                    return False
                if idx_set.count(pos[0]) > 0 or idx_set.count(pos[1]) > 0:
                    # the boundary should be unique
                    return False
                idx_set.append(pos[0])
                idx_set.append(pos[1])
            previous_recompute_location = None
            for idx in range(len(recompute_locations)):
                # recompute block and backward block should appear alternating
                if previous_recompute_location is not None:
                    if (
                        backward_positions[idx][1]
                        >= previous_recompute_location[0]
                    ):
                        # backward ops overlap with recompute ops
                        return False
                if backward_positions[idx][0] <= recompute_locations[idx][1]:
                    # backward ops overlap with recompute ops
                    return False
                previous_recompute_location = recompute_locations[idx]
        return True

    def _apply_single_impl(self, main_program, startup_program, context):
        logging.info("apply sp opt pass")
        self._previous_passes = context.passes
        self._recompute = False
        self._gradient_merge = False
        for previous_pass in self._previous_passes:
            if previous_pass.name == 'auto_parallel_recompute':
                logging.info("sp pass with recompute")
                self._recompute = True
            if previous_pass.name == 'auto_parallel_gradient_merge_pass':
                logging.info("sp pass with gradient merge")
                self._gradient_merge = True
        self.local_rank = self._attrs["global_rank"]
        # step 1: get mp and pp degree, determine whether is pp first/last stage
        self._parse_hybrid_degree(startup_program)

        if self.mp_degree <= 1:
            logging.warning(
                "The sp opt pass got invalid mp degree: {} the pass won't apply.".format(
                    self.mp_degree
                ),
            )
            return

        # step 2: locate the transformer blocks
        transformer_block_location = self._locate_transformer_block(
            main_program
        )
        forward_location = transformer_block_location["forward"]
        backward_location = transformer_block_location["backward"]
        recompute_location = transformer_block_location["recompute"]
        self._recompute_locations = recompute_location
        assert len(forward_location) == len(backward_location)
        if len(backward_location) == 0:
            logging.info(
                "cannot detect any transformer block, sp pass will exit"
            )
            return
        backward_location = sorted(
            backward_location, key=lambda x: x[0], reverse=True
        )
        forward_location = sorted(
            forward_location, key=lambda x: x[0], reverse=True
        )
        if self._recompute:
            assert len(forward_location) == len(recompute_location)
            recompute_location = sorted(
                recompute_location, key=lambda x: x[0], reverse=True
            )
        assert self._location_validation_check(
            forward_location, backward_location, recompute_location
        )

        # step 3: Get batch size, sequence len, hidden size
        self._get_hyper_params(main_program, forward_location[-1][0])

        # step 4: Replace ops for one backward and one recompute.
        # Traverse backward to maintain the relative ops' position.
        for idx, locations in enumerate(backward_location):
            # Only if current device is the first pp stage and
            # current backward transformer block is the last one
            # can this block be the first block in the origin single card program.
            first = self.is_pp_first_stage and idx == 0
            # Only if current device is the last pp stage and
            # current backward transformer block is the first one
            # can this block be the last block in the origin single card program.
            last = self.is_pp_last_stage and idx == len(backward_location) - 1
            main_program = self._backward_transformer_block_replace(
                main_program, locations[0], locations[1], first, last
            )
            if self._recompute:
                main_program = self._forward_transformer_block_replace(
                    main_program,
                    recompute_location[idx][0],
                    recompute_location[idx][1],
                    first,
                    last,
                    for_recompute=True,
                )

        # step 5: Replace ops for forward.
        # Traverse backward to maintain the relative ops' position.
        for idx, locations in enumerate(forward_location):
            # Only if current device is the first pp stage and
            # current forward transformer block is the first one
            # can this block be the first block in the origin single card program.
            first = self.is_pp_first_stage and idx == len(forward_location) - 1
            # Only if current device is the last pp stage and
            # current forward transformer block is the last one
            # can this block be the last block in the origin single card program.
            last = self.is_pp_last_stage and idx == 0
            main_program = self._forward_transformer_block_replace(
                main_program, locations[0], locations[1], first, last
            )

        # step 6: Use the updated var (created in the forward pass) to replace origin var in the backward.
        main_program = self._backward_input_replacing(main_program)

        # step 7: insert extra allreduce sum for grad need sync across all mp stages
        main_program = self._insert_allreduce(main_program)

        # step 8: rerun infer shape to update all tensors' shape
        main_program = self._infer_shape_updadte(main_program)

        logging.info(
            f"sequence parallel pass has replaced {len(forward_location)} transformer blocks"
        )

        return main_program
