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

from .pass_base import PassBase, register_pass
from paddle.fluid import core, unique_name
from paddle.fluid.framework import Variable, Operator
from paddle.distributed.auto_parallel.process_mesh import ProcessMesh
from paddle.distributed.auto_parallel.utils import get_loss_op, set_var_dist_attr, print_program_with_dist_attr
from paddle.distributed.auto_parallel.utils import naive_set_dist_op_attr_for_program_by_mesh_and_mapping
from paddle.distributed.auto_parallel.dist_attribute import OperatorDistributedAttribute
from paddle.fluid.backward import ProgramStats, _append_grad_suffix_, _get_no_grad_set_name
from paddle.fluid.backward import _find_no_grad_vars, _rename_arg_, _find_op_path_, _add_needed_descs_to_block


class RecomputeState(ProgramStats):
    def __init__(self, block, ops):
        super(RecomputeState, self).__init__(block=block, ops=ops)
        self._block = block
        self._ops = ops
        self.var_op_deps = {}

    @property
    def loss_op_index(self):
        return self._loss_op_index

    def _sync_loss_op_index(self):
        loss_op = get_loss_op(self._block)
        self._loss_op_index = _find_op_index(self._block, loss_op)

    def build_stats(self):
        for i, op in enumerate(self._ops):
            for name in op.desc.input_arg_names():
                if name in self.var_op_deps:
                    self.var_op_deps[name]["var_as_input_ops"].extend([i])
                else:
                    self.var_op_deps[name] = {}
                    self.var_op_deps[name]["var_as_input_ops"] = [i]
                    self.var_op_deps[name]["var_as_output_ops"] = []

            for name in op.desc.output_arg_names():
                if name in self.var_op_deps:
                    self.var_op_deps[name]["var_as_output_ops"].extend([i])
                else:
                    self.var_op_deps[name] = {}
                    self.var_op_deps[name]["var_as_input_ops"] = []
                    self.var_op_deps[name]["var_as_output_ops"] = [i]

    def get_recompute_segments(self, checkpoints):
        segments = []
        start_idx = -1
        pre_segment_end_idx = -1
        while start_idx + 1 < len(checkpoints):
            if start_idx == -1:
                ckpt_name = checkpoints[start_idx + 1]
                if ckpt_name not in self.var_op_deps:
                    start_idx += 1
                    continue
                op_idx_list = self.var_op_deps[ckpt_name]["var_as_output_ops"]
                if op_idx_list:
                    segments.append([0, max(op_idx_list) + 1])
            else:
                flag, min_idx, max_idx = self.is_subgraph(
                    [checkpoints[start_idx]], [checkpoints[start_idx + 1]])
                if flag:
                    min_idx = self._update_segment_start(min_idx,
                                                         pre_segment_end_idx)
                    segments.append([min_idx, max_idx + 1])
                else:
                    logging.info("Could not recompute op range [{}] - [{}] ".
                                 format(min_idx, max_idx + 1))

            start_idx += 1

        return segments

    def modify_forward_desc_for_recompute(self, dist_context):
        op_types = [op.desc.type() for op in self._ops]
        if "dropout" not in op_types:
            return

        op_idx = 0
        while op_idx < len(self._ops):
            cur_op = self._ops[op_idx]
            if "grad" in cur_op.type:
                break
            if cur_op.type != "dropout":
                op_idx += 1
                continue
            if cur_op.input("Seed") is not None and len(cur_op.input("Seed")):
                op_idx += 1
                continue

            cur_op_dist_attr = dist_context.get_op_dist_attr_for_program(cur_op)
            # insert seed op to guarantee that two dropout op have some outputs
            op_unique_name = unique_name.generate("seed")
            var_unique_name = unique_name.generate_with_ignorable_key(".".join(
                [op_unique_name, 'tmp']))
            seed_var = self._block.create_var(
                name=var_unique_name,
                dtype='int32',
                type=core.VarDesc.VarType.LOD_TENSOR,
                persistable=False,
                stop_gradient=False)

            # set new seed_var's distributed attribution
            ref_dims_mapping = [-1]
            ref_process_mesh = cur_op_dist_attr.process_mesh
            seed_var_dist_attr = set_var_dist_attr(
                dist_context, seed_var, ref_dims_mapping, ref_process_mesh)

            seed = 0 if cur_op.attr("fix_seed") is False else int(
                cur_op.attr("seed"))
            seed_op = self._block._insert_op_without_sync(
                index=cur_op.idx,
                type="seed",
                inputs={},
                outputs={"Out": seed_var},
                attrs={"seed": seed,
                       "force_cpu": True})

            # set new seed op's distributed attribution
            naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
                seed_op, ref_process_mesh, ref_dims_mapping, dist_context)

            # modify dropout op desc
            self._ops.insert(op_idx, seed_op)
            cur_op.desc.set_input("Seed", [var_unique_name])
            cur_op.desc.remove_attr("fix_seed")
            cur_op.desc.remove_attr("seed")
            cur_op_dist_attr.set_input_dist_attr(seed_var.name,
                                                 seed_var_dist_attr)
            self._block._sync_with_cpp()
            op_idx += 2

        self._sync_loss_op_index()


def _get_op_by_id(ops, id):
    for op in ops:
        if op.desc.id() == id:
            return op
    return None


def _find_op_index(block, cur_op):
    for idx in range(block.desc.op_size()):
        if cur_op.desc == block.desc.op(idx):
            return idx
    return -1


def _get_stop_gradients(program, no_grad_set):
    if no_grad_set is None:
        no_grad_set = set()
    else:
        no_grad_set = _get_no_grad_set_name(no_grad_set)

    no_grad_set_name = set()
    for var in program.list_vars():
        assert isinstance(var, Variable)
        if "@GRAD" in var.name:
            break
        if var.stop_gradient:
            no_grad_set_name.add(_append_grad_suffix_(var.name))
    no_grad_set_name.update(list(map(_append_grad_suffix_, no_grad_set)))
    return no_grad_set_name


def reset_op_dist_attr(dst_op, src_op_dist_attr, var_dist_attr_dict):

    dst_op_dist_attr = OperatorDistributedAttribute()

    for input in dst_op.desc.input_arg_names():
        if input in var_dist_attr_dict:
            in_dims_mapping = var_dist_attr_dict[input].dims_mapping
        else:
            in_dims_mapping = src_op_dist_attr.get_input_dims_mapping(input)
        dst_op_dist_attr.set_input_dims_mapping(input, in_dims_mapping)
    for output in dst_op.desc.output_arg_names():
        if output in var_dist_attr_dict:
            out_dims_mapping = var_dist_attr_dict[output].dims_mapping
        else:
            out_dims_mapping = src_op_dist_attr.get_output_dims_mapping(output)
        dst_op_dist_attr.set_output_dims_mapping(output, out_dims_mapping)
    dst_op_dist_attr.process_mesh = src_op_dist_attr.process_mesh
    return dst_op_dist_attr


@register_pass("auto_parallel_recompute")
class RecomputePass(PassBase):
    def __init__(self):
        super(RecomputePass, self).__init__()
        self.set_attr("checkpoints", None)
        self.set_attr("dist_context", None)
        self.set_attr("loss", None)
        self.set_attr("parameter_list", None)
        self.set_attr("no_grad_set", None)

    def _check_self(self):
        if self.get_attr("dist_context") is None:
            return False
        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_programs, startup_programs, context):
        checkpoints = self.get_attr("checkpoints")
        assert checkpoints is not None, "Checkpoints should be set in advance."

        # find loss op
        main_block = main_programs.global_block()
        loss_op = get_loss_op(main_block)
        loss = main_block.var(loss_op.output_arg_names[0])

        # get no grad var name
        no_grad_set = self.get_attr("no_grad_set")
        no_grad_set_name = _get_stop_gradients(main_programs, no_grad_set)

        # get op_path which is related to loss
        op_path = _find_op_path_(main_block, [loss], [], no_grad_set_name)
        # get no_grad_vars which is not related to loss
        # no_grad_vars = _find_no_grad_vars(main_block, op_path, [loss], no_grad_set_name)
        # no_grad_set_name.update(list(map(_append_grad_suffix_, no_grad_vars)))

        self._dist_context = self.get_attr("dist_context")
        dist_op_context = self._dist_context.dist_op_context
        recompute_state = RecomputeState(main_block, op_path)
        recompute_state.modify_forward_desc_for_recompute(self._dist_context)
        recompute_state.build_stats()
        checkpoints = recompute_state.sort_checkpoints(checkpoints)
        segments = recompute_state.get_recompute_segments(checkpoints)

        # print("=======================after insert seed op=======================")
        # print_program_with_dist_attr(main_programs, self._dist_context)

        for i, (idx1, idx2) in enumerate(segments):
            logging.info("recompute segment[{}]".format(i))
            logging.info("segment start op: [{}]: [{}]".format(op_path[
                idx1].desc.type(), op_path[idx1].desc.input_arg_names()))
            logging.info("segment end op: [{}]: [{}]".format(op_path[
                idx2 - 1].desc.type(), op_path[idx2 - 1].desc.input_arg_names(
                )))
            logging.info("recompute segment[{}]".format(i))
            logging.info("segment start op: [{}]: [{}]".format(op_path[
                idx1].desc.type(), op_path[idx1].desc.input_arg_names()))
            logging.info("segment end op: [{}]: [{}]".format(op_path[
                idx2 - 1].desc.type(), op_path[idx2 - 1].desc.input_arg_names(
                )))

        vars_should_be_hold = []
        for segment in segments:
            vars_should_be_hold.extend(
                recompute_state.get_out_of_subgraph_vars(segment[0], segment[
                    1]))
        cross_vars = set(vars_should_be_hold) - set(checkpoints)
        logging.info(
            "found [{}] vars which cross recompute segment: [{}], better checkpoints might be set to reduce those vars".
            format(len(cross_vars), cross_vars))

        vars_should_be_hold.extend(recompute_state.get_reserved_vars())
        vars_should_be_hold.extend(recompute_state.get_input_nodes())
        vars_should_be_hold = list(set(vars_should_be_hold))
        vars_in_memory = vars_should_be_hold + checkpoints

        var_name_dict = {}
        ckpt_ops_dict = {}
        var_dist_attr_dict = {}
        op_dist_attr_dict = {}
        buffer_block = main_block.program._create_block()

        if segments == []:
            return

        for i, segment in enumerate(segments[::-1]):
            ckpt2ops = {}
            fwd_ops = op_path[segment[0]:segment[1]]
            var_suffix = ".subprog_%d" % i
            for op in fwd_ops:
                input_and_output_names = []
                input_and_output_names.extend(op.desc.input_arg_names())
                input_and_output_names.extend(op.desc.output_arg_names())
                cur_op_dist_attr = self._dist_context.get_op_dist_attr_for_program(
                    op)
                op_dist_attr_dict[op.desc.id()] = cur_op_dist_attr
                for name in input_and_output_names:
                    if main_block.var(name).persistable or name in checkpoints:
                        continue
                    if name in vars_should_be_hold:
                        continue
                    if name not in var_name_dict:
                        ref_process_mesh = cur_op_dist_attr.process_mesh
                        if name in op.desc.input_arg_names():
                            ref_dims_mapping = cur_op_dist_attr.get_input_dims_mapping(
                                name)
                        else:
                            ref_dims_mapping = cur_op_dist_attr.get_output_dims_mapping(
                                name)
                        var_name_dict[name] = name + var_suffix
                        ref_var = main_block.var(name)
                        rc_var = main_block.create_var(
                            name=var_name_dict[name],
                            shape=ref_var.shape,
                            dtype=ref_var.dtype,
                            type=ref_var.type,
                            persistable=ref_var.persistable,
                            stop_gradient=ref_var.stop_gradient)
                        # set new recompute var's dist attr
                        rc_var_dist_attr = set_var_dist_attr(
                            self._dist_context, rc_var, ref_dims_mapping,
                            ref_process_mesh)
                        var_dist_attr_dict[var_name_dict[
                            name]] = rc_var_dist_attr

            buffer_descs = _add_needed_descs_to_block(
                fwd_ops, buffer_block, main_block, vars_in_memory)
            for key in var_name_dict:
                _rename_arg_(buffer_descs, key, var_name_dict[key])

            ckpt2ops[0] = 0
            ckpt2ops[1] = buffer_descs
            ckpt_op = op_path[segment[1] - 1]
            ckpt_ops_dict[ckpt_op.desc.id()] = ckpt2ops

        ops = main_block.ops
        loss_op_idx = recompute_state.loss_op_index
        for i in range(len(ops) - 1, loss_op_idx, -1):
            grad_op = ops[i]
            if grad_op.type == "dropout_grad":
                grad_op.desc.remove_attr("fix_seed")
                grad_op.desc.remove_attr("seed")
                main_block._sync_with_cpp()

            for key in var_name_dict:
                _rename_arg_([grad_op.desc], key, var_name_dict[key])
                grad_op_dist_attr = self._dist_context.get_op_dist_attr_for_program(
                    grad_op)
                new_grad_op_dist_attr = reset_op_dist_attr(
                    grad_op, grad_op_dist_attr, var_dist_attr_dict)
                self._dist_context.set_op_dist_attr_for_program(
                    grad_op, new_grad_op_dist_attr)

            if grad_op.desc.id() in dist_op_context.gradopidx2opidx:
                fwd_op = _get_op_by_id(
                    ops[:loss_op_idx + 1],
                    dist_op_context.gradopidx2opidx[grad_op.desc.id()])
                assert fwd_op is not None

                fwd_op_id = dist_op_context.gradopidx2opidx[grad_op.desc.id()]
                if fwd_op_id in ckpt_ops_dict and ckpt_ops_dict[fwd_op_id][
                        0] == 0:
                    idx = grad_op.idx
                    while idx - 1 >= 0 and ops[idx - 1].type == "sum":
                        idx -= 1
                    descs = ckpt_ops_dict[fwd_op_id][1]
                    for _, op_desc in reversed(list(enumerate(descs))):
                        rc_desc = main_block.desc._insert_op(idx)
                        rc_desc.copy_from(op_desc)
                        rc_op = Operator(main_block, rc_desc)
                        main_block.ops.insert(idx, rc_op)

                        op_dist_attr = op_dist_attr_dict[op_desc.id()]
                        rc_op_dist_attr = reset_op_dist_attr(
                            rc_op, op_dist_attr, var_dist_attr_dict)
                        self._dist_context.set_op_dist_attr_for_program(
                            rc_op, rc_op_dist_attr)

                    ckpt_ops_dict[fwd_op_id][0] = 1

                    main_block._sync_with_cpp()

        main_programs._sync_with_cpp()
        print("***********************final program***********************")
        print_program_with_dist_attr(main_programs, self._dist_context)
