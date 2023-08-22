# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.distributed.fleet.meta_optimizers.common import OP_ROLE_KEY, OpRole
from paddle.fluid.backward import (
    ProgramStats,
    _append_grad_suffix_,
    _find_op_path_,
    _get_no_grad_set_name,
    _rename_arg_,
)
from paddle.framework import core
from paddle.utils import unique_name

from ..auto_parallel.static.dist_attribute import OperatorDistAttr
from ..auto_parallel.static.utils import (
    get_loss_op,
    insert_dependencies_for_two_ops,
    is_backward_op,
    is_recompute_op,
    naive_set_dist_op_attr_for_program_by_mesh_and_mapping,
    set_dist_op_desc_original_id,
    set_var_dist_attr,
)
from .pass_base import PassBase, register_pass


class RecomputeState(ProgramStats):
    def __init__(self, block, ops):
        super().__init__(block=block, ops=ops)
        self.seg_op_deps = {}
        self._checkpoints = []
        self._reserved_vars = []

    @property
    def checkpoints(self):
        return self._checkpoints

    @property
    def reserved_vars(self):
        return self._reserved_vars

    def is_recompute(self):
        return any([is_recompute_op(op) for op in self.ops])

    def build_states(self):
        for i, op in enumerate(self.ops):
            if is_backward_op(op):
                break

            for name in op.input_arg_names:
                if name in self.var_op_deps:
                    self.var_op_deps[name]["var_as_input_ops"].extend([i])
                else:
                    self.var_op_deps[name] = {}
                    self.var_op_deps[name]["var_as_input_ops"] = [i]
                    self.var_op_deps[name]["var_as_output_ops"] = []

            for name in op.output_arg_names:
                if name in self.var_op_deps:
                    self.var_op_deps[name]["var_as_output_ops"].extend([i])
                else:
                    self.var_op_deps[name] = {}
                    self.var_op_deps[name]["var_as_input_ops"] = []
                    self.var_op_deps[name]["var_as_output_ops"] = [i]

            if not is_recompute_op(op):
                self._checkpoints.extend(op.output_arg_names)
                continue

            seg_name = op.attr('op_namescope')
            if seg_name not in self.seg_op_deps:
                self.seg_op_deps[seg_name] = [i]
            else:
                assert (
                    self.seg_op_deps[seg_name][-1] + 1 == i
                ), "The recompute segment's ops should be continuous"
                self.seg_op_deps[seg_name].extend([i])

    def get_recompute_segments(self, no_recompute_segments=[]):
        segments = []
        for segment_idx in self.seg_op_deps.values():
            if len(segment_idx) == 1:
                continue
            segments.append([segment_idx[0], segment_idx[-1] + 1])
            self._checkpoints.extend(self.ops[segment_idx[-1]].output_arg_names)

        for i in sorted(no_recompute_segments, reverse=True):
            assert i < len(
                segments
            ), "the no_recompute_segments idx [{}] should be lower the number of segment [{}]".format(
                i, len(segments)
            )
            segments.pop(i)

        return segments

    def modify_forward_desc_for_recompute(self, dist_context):
        """
        If program's foward part has 'dropout' op, this function will insert
        a seed op before it to guarantee that two dropout op have the same outputs.
        """
        op_types = [op.type for op in self.ops]
        if "dropout" not in op_types and "fused_dropout_add" not in op_types:
            return

        op_idx = 0
        while op_idx < len(self.ops):
            cur_op = self.ops[op_idx]
            if "grad" in cur_op.type:
                break
            if cur_op.type == "seed":
                self._reserved_vars.extend(cur_op.output_arg_names)
                op_idx += 1
                continue
            if cur_op.type not in ["dropout", "fused_dropout_add"]:
                op_idx += 1
                continue
            seed_tensor_name = (
                "seed_tensor" if cur_op.type == "fused_dropout_add" else "Seed"
            )
            if cur_op.input(seed_tensor_name) is not None and len(
                cur_op.input(seed_tensor_name)
            ):
                op_idx += 1
                continue

            cur_op_dist_attr = dist_context.get_op_dist_attr_for_program(cur_op)
            # insert seed op to guarantee that two dropout op have the same outputs
            # NOTE Hack for adopt recompute for random control, for more info see dist_dropout.py
            # new seed added by recompute should have a prefix to distinguish with seed added by user or other moudule.
            op_unique_name = unique_name.generate("rc_seed")
            var_unique_name = unique_name.generate_with_ignorable_key(
                ".".join([op_unique_name, 'tmp'])
            )
            self._reserved_vars.append(var_unique_name)
            seed_var = self.block.create_var(
                name=var_unique_name,
                dtype='int32',
                type=core.VarDesc.VarType.LOD_TENSOR,
                persistable=False,
                stop_gradient=False,
            )

            # set new seed_var's dist_attr
            ref_dims_mapping = [-1]
            ref_process_mesh = cur_op_dist_attr.process_mesh
            seed_var_dist_attr = set_var_dist_attr(
                dist_context, seed_var, ref_dims_mapping, ref_process_mesh
            )

            seed = (
                0
                if cur_op.attr("fix_seed") is False
                else int(cur_op.attr("seed"))
            )
            # TODO add dependency for seed op to ensure it be issued just before recompute.
            seed_op = self.block._insert_op_without_sync(
                index=cur_op.idx,
                type="seed",
                inputs={},
                outputs={"Out": seed_var},
                attrs={"seed": seed, "force_cpu": True},
            )
            seed_op._set_attr('op_namescope', cur_op.attr('op_namescope'))
            # set new seed op's dist_attr
            naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
                seed_op, ref_process_mesh, ref_dims_mapping, dist_context
            )

            # modify dropout op's desc
            self.ops.insert(op_idx, seed_op)
            cur_op.desc.set_input(seed_tensor_name, [var_unique_name])
            cur_op._remove_attr("fix_seed")
            cur_op._remove_attr("seed")
            cur_op_dist_attr.set_input_dist_attr(
                seed_var.name, seed_var_dist_attr
            )
            op_idx += 2

        self.block._sync_with_cpp()


def _find_op_index(block, cur_op):
    for idx in range(block.desc.op_size()):
        if cur_op.desc == block.desc.op(idx):
            return idx
    return -1


def _get_stop_gradients(program, no_grad_set=None):
    """get no grad var"""
    if no_grad_set is None:
        no_grad_set = set()
    else:
        no_grad_set = _get_no_grad_set_name(no_grad_set)

    no_grad_set_name = set()
    for var in program.list_vars():
        if "@GRAD" in var.name:
            break
        if var.stop_gradient:
            no_grad_set_name.add(_append_grad_suffix_(var.name))
    no_grad_set_name.update(list(map(_append_grad_suffix_, no_grad_set)))
    return no_grad_set_name


def _add_needed_descs_to_block(
    descs, block, main_block, vars_should_be_hold, dist_context
):
    """
    Get the recomputed ops which will insert the backward part
    """
    if len(descs) == 0:
        return []

    result_descs = []
    for desc in descs:
        # if isinstance(desc, framework.Operator):
        if isinstance(desc, paddle.static.Operator):
            desc = desc.desc
        if isinstance(desc, tuple):
            desc = desc[0]
        is_needed = False
        for name in desc.output_arg_names():
            if main_block.has_var(name) and main_block.var(name).persistable:
                continue
            if name not in vars_should_be_hold:
                is_needed = True
        if is_needed:
            new_op_desc = block.desc.append_op()
            new_op_desc.copy_from(desc)
            set_dist_op_desc_original_id(new_op_desc, desc, dist_context)
            new_op_desc._set_attr(OP_ROLE_KEY, OpRole.Backward)
            result_descs.append(new_op_desc)
    return result_descs


def _find_op_path(main_program, loss, no_grad_set=None):
    no_grad_set_name = _get_stop_gradients(main_program, no_grad_set)
    op_path = _find_op_path_(
        main_program.global_block(), [loss], [], no_grad_set_name
    )
    return op_path


@register_pass("auto_parallel_recompute")
class RecomputePass(PassBase):
    def __init__(self):
        super().__init__()
        self.set_attr("loss", None)
        self.set_attr("dist_context", None)
        self.set_attr("no_grad_set", None)
        self.set_attr("no_recompute_segments", [])

    def _check_self(self):
        if self.get_attr("dist_context") is None:
            return False
        if self.get_attr("loss") is None:
            return False
        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, context):
        loss = self.get_attr("loss")
        no_grad_set = self.get_attr("no_grad_set")
        no_recompute_segments = self.get_attr("no_recompute_segments")
        self._dist_context = self.get_attr("dist_context")

        # 0. get op_path which is related to loss
        main_block = main_program.global_block()
        op_path = _find_op_path(main_program, loss, no_grad_set)

        # 1. build recompute state
        rc_state = RecomputeState(main_block, op_path)
        if not rc_state.is_recompute():
            return

        # 2. get the segments to be recomputed
        rc_state.modify_forward_desc_for_recompute(self._dist_context)
        rc_state.build_states()
        segments = rc_state.get_recompute_segments(no_recompute_segments)
        if segments == []:
            return

        for i, (idx1, idx2) in enumerate(segments):
            logging.info(f"recompute segment[{i + 1}/{len(segments)}]")
            logging.info(
                "segment start op: [{}]: [{}] [{}]".format(
                    rc_state.ops[idx1].type,
                    rc_state.ops[idx1].input_arg_names,
                    rc_state.ops[idx1].output_arg_names,
                )
            )
            logging.info(
                "segment end op: [{}]: [{}] [{}]".format(
                    rc_state.ops[idx2 - 1].type,
                    rc_state.ops[idx2 - 1].input_arg_names,
                    rc_state.ops[idx2 - 1].output_arg_names,
                )
            )

        # 3. get vars that should be hold in memory
        vars_should_be_hold = []
        for segment in segments:
            vars_should_be_hold.extend(
                rc_state.get_out_of_subgraph_vars(segment[0], segment[1])
            )
        cross_vars = set(vars_should_be_hold) - set(rc_state.checkpoints)
        logging.info(
            "found [{}] vars which cross recompute segment: [{}],"
            "better checkpoints might be set to reduce those vars".format(
                len(cross_vars), cross_vars
            )
        )
        vars_should_be_hold.extend(rc_state.reserved_vars)
        vars_should_be_hold.extend(rc_state.get_input_nodes())
        vars_should_be_hold = list(
            set(vars_should_be_hold) | set(rc_state.checkpoints)
        )

        # 4. get the fwd ops desc to be recomputed.
        var_name_dict = {}  # varname --> varname.subprog_XXX
        ckpt_ops_dict = {}  # ckpt_op_id --> segment_descs
        buffer_block = main_block.program._create_block()
        for i, segment in enumerate(segments[::-1]):
            fwd_ops = op_path[segment[0] : segment[1]]
            var_suffix = ".subprog_%d" % i
            for op in fwd_ops:
                input_and_output_names = []
                input_and_output_names.extend(op.input_arg_names)
                input_and_output_names.extend(op.output_arg_names)

                cur_op_dist_attr = (
                    self._dist_context.get_op_dist_attr_for_program(op)
                )
                assert cur_op_dist_attr is not None

                for name in input_and_output_names:
                    if (
                        main_block.var(name).persistable
                        or name in vars_should_be_hold
                    ):
                        continue
                    if name not in var_name_dict:
                        ref_process_mesh = cur_op_dist_attr.process_mesh
                        if name in op.input_arg_names:
                            ref_dims_mapping = (
                                cur_op_dist_attr.get_input_dims_mapping(name)
                            )
                        else:
                            ref_dims_mapping = (
                                cur_op_dist_attr.get_output_dims_mapping(name)
                            )

                        # record recomputed var's old_name and new_name (old_name.subprog_XXX)
                        # create new var with new name
                        var_name_dict[name] = name + var_suffix
                        ref_var = main_block.var(name)
                        rc_var = main_block.create_var(
                            name=var_name_dict[name],
                            shape=ref_var.shape,
                            dtype=ref_var.dtype,
                            type=ref_var.type,
                            persistable=ref_var.persistable,
                            stop_gradient=ref_var.stop_gradient,
                        )
                        # set new recomputed var's dist attr
                        set_var_dist_attr(
                            self._dist_context,
                            rc_var,
                            ref_dims_mapping,
                            ref_process_mesh,
                        )
            # get recomputed segment's descs
            segment_descs = _add_needed_descs_to_block(
                fwd_ops,
                buffer_block,
                main_block,
                vars_should_be_hold,
                self._dist_context,
            )
            # rename recomputed ops' input and output var name
            for key in var_name_dict:
                _rename_arg_(segment_descs, key, var_name_dict[key])

            # NOTE: one forward op could be correspond to multiple xxx_grad op.
            # When traversing all grad_ops in reverse, need to set a flag to indicate
            # whether the ckpt and its segment_descs can be used.
            ckpt_op = op_path[segment[1] - 1]
            ckpt_ops_dict[ckpt_op.desc.original_id()] = [True, segment_descs]

        # 5. insert recomputed fwd ops into backward parse
        ops = main_block.ops
        loss_op = get_loss_op(main_block)
        loss_op_idx = _find_op_index(main_block, loss_op)
        dist_op_context = self._dist_context.dist_op_context
        assert loss_op_idx != -1
        # Traversing all grad_ops in reverse, and if the fwd op corresponding to reverse op is checkpoints,
        # segments ops should be inserted.
        for i in range(len(ops) - 1, loss_op_idx, -1):
            grad_op = ops[i]
            # remove some attrs of dropout_grad op's desc
            if grad_op.type == "dropout_grad":
                grad_op._remove_attr("fix_seed")
                grad_op._remove_attr("seed")

            input_and_output_names = []
            input_and_output_names.extend(grad_op.input_arg_names)
            input_and_output_names.extend(grad_op.output_arg_names)

            for varname in var_name_dict:
                if varname not in input_and_output_names:
                    continue
                self.reset_op_dist_attr(grad_op, var_name_dict)
                _rename_arg_([grad_op.desc], varname, var_name_dict[varname])

            # insert recomputed ops
            original_id = grad_op.desc.original_id()
            if original_id in dist_op_context.grad_op_id_to_op_id:
                fwd_op_id = dist_op_context.grad_op_id_to_op_id[original_id]
                if fwd_op_id in ckpt_ops_dict and ckpt_ops_dict[fwd_op_id][0]:
                    idx = grad_op.idx
                    while idx - 1 >= 0 and ops[idx - 1].type == "sum":
                        idx -= 1
                    segment_descs = ckpt_ops_dict[fwd_op_id][1]
                    rc_op = None
                    for _, op_desc in reversed(list(enumerate(segment_descs))):
                        rc_op = main_block._insert_op_without_sync(
                            idx, type='nop'
                        )
                        rc_desc = rc_op.desc
                        rc_desc.copy_from(op_desc)
                        rc_desc.set_original_id(rc_desc.id())
                        # set recomputed ops' dist attr
                        fwd_op_dist_attr = self._dist_context.get_op_dist_attr_for_program_with_id(
                            op_desc.original_id()
                        )
                        assert fwd_op_dist_attr is not None
                        self.set_op_dist_attr(
                            rc_op, fwd_op_dist_attr, var_name_dict
                        )

                    ckpt_ops_dict[fwd_op_id][0] = False
                    if rc_op:
                        prior_op = main_block.ops[rc_op.idx - 1]
                        posterior_op = rc_op
                        prior_mesh = (
                            self._dist_context.get_op_dist_attr_for_program(
                                prior_op
                            ).process_mesh
                        )
                        posterior_mesh = (
                            self._dist_context.get_op_dist_attr_for_program(
                                posterior_op
                            ).process_mesh
                        )
                        # NOTE if two recompute segements across two pipeline stages
                        # not need dependecies for it
                        if prior_mesh == posterior_mesh:
                            insert_dependencies_for_two_ops(
                                main_block,
                                idx,
                                prior_op,
                                posterior_op,
                                self._dist_context,
                                is_recompute=True,
                                sync=False,
                                op_namescope="recompute_segment_dep",
                            )
        main_program._sync_with_cpp()

    def reset_op_dist_attr(self, op, var_name_dict):
        op_dist_attr = self._dist_context.get_op_dist_attr_for_program(op)
        assert op_dist_attr is not None
        for input in op.input_arg_names:
            if input in var_name_dict.keys():
                in_dist_attr = op_dist_attr.get_input_dist_attr(input)
                op_dist_attr.set_input_dist_attr(
                    var_name_dict[input], in_dist_attr
                )
        for output in op.output_arg_names:
            if output in var_name_dict.keys():
                out_dist_attr = op_dist_attr.get_output_dist_attr(output)
                op_dist_attr.set_output_dist_attr(
                    var_name_dict[output], out_dist_attr
                )

    def set_op_dist_attr(self, op, old_dist_attr, var_name_dict):
        new_dist_attr = OperatorDistAttr()
        new_dist_attr.is_recompute = True
        new_dist_attr.impl_idx = old_dist_attr.impl_idx
        new_dist_attr.impl_type = old_dist_attr.impl_type
        new_dist_attr.process_mesh = old_dist_attr.process_mesh
        for input in old_dist_attr.inputs_dist_attrs.keys():
            if input in var_name_dict.keys():
                in_dist_attr = old_dist_attr.inputs_dist_attrs[input]
                new_dist_attr.set_input_dist_attr(
                    var_name_dict[input], in_dist_attr
                )
            else:
                in_dist_attr = old_dist_attr.inputs_dist_attrs[input]
                new_dist_attr.set_input_dist_attr(input, in_dist_attr)
        for output in old_dist_attr.outputs_dist_attrs.keys():
            if output in var_name_dict.keys():
                out_dist_attr = old_dist_attr.outputs_dist_attrs[output]
                new_dist_attr.set_output_dist_attr(
                    var_name_dict[output], out_dist_attr
                )
            else:
                out_dist_attr = old_dist_attr.outputs_dist_attrs[output]
                new_dist_attr.set_output_dist_attr(output, out_dist_attr)
        self._dist_context.set_op_dist_attr_for_program(op, new_dist_attr)
