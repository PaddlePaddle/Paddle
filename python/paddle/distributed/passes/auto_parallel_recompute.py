# # Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# # 
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# # 
# #     http://www.apache.org/licenses/LICENSE-2.0
# # 
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# import paddle
# import logging

# from .pass_base import PassBase, register_pass
# from paddle.fluid import core, unique_name
# from paddle.fluid.framework import Variable
# from paddle.fluid.data_feeder import check_variable_and_dtype, check_type
# from paddle.distributed.auto_parallel.process_mesh import ProcessMesh
# from paddle.distributed.auto_parallel.utils import is_forward_op, is_backward_op
# from paddle.distributed.auto_parallel.utils import get_loss_op, set_var_dist_attr
# from paddle.distributed.auto_parallel.dist_attribute import OperatorDistributedAttribute
# from paddle.fluid.backward import _append_grad_suffix_, _get_no_grad_set_name, _get_output_names, _some_in_set_
# from paddle.fluid.backward import _create_loss_op_desc_, _find_no_grad_vars, _find_loss_op_

# class RecomputeState:
#     def __init__(self, block, ops):
#         self._block = block
#         self._ops = ops
#         self._var_op_deps = {}

#     def _build_var_op_dependency(self):
#         for i, op in enumerate(self._ops):
#             for name in op.desc.input_arg_names():
#                 if name in self._var_op_deps:
#                     self._var_op_deps[name]["var_as_input_ops"].extend([i])
#                 else:
#                     self._var_op_deps[name] = {}
#                     self._var_op_deps[name]["var_as_input_ops"] = [i]
#                     self._var_op_deps[name]["var_as_output_ops"] = []

#             for name in op.desc.output_arg_names():
#                 if name in self._var_op_deps:
#                     self._var_op_deps[name]["var_as_output_ops"].extend([i])
#                 else:
#                     self._var_op_deps[name] = {}
#                     self._var_op_deps[name]["var_as_input_ops"] = []
#                     self._var_op_deps[name]["var_as_output_ops"] = [i]

#     def _sort_checkpoints(self, checkpoints):
#         sorted_checkpoints = []
#         for ckpt_name in checkpoints:
#             if ckpt_name not in self._var_op_deps:
#                 logging.info(
#                     "Delete %s from checkpoints, because it is not used in paddle program."
#                     % ckpt_name)
#             elif self._var_op_deps[ckpt_name]["var_as_output_ops"] == []:
#                 sorted_checkpoints.append((ckpt_name, -1))
#             else:
#                 sorted_checkpoints.append((ckpt_name, max(self._var_op_deps[ckpt_name]["var_as_output_ops"])))
#         sorted_checkpoints = sorted(sorted_checkpoints, key=lambda x: x[1])
#         checkpoints = [ckpt[0] for ckpt in sorted_checkpoints]
#         return checkpoints

#     def _is_segment(self, checkpoint_1, checkpoint_2):
#         if checkpoint_1 not in self._var_op_deps or checkpoint_2 not in self._var_op_deps:
#             return False
#         op_idx_list_1 = self._var_op_deps[checkpoint_1]["var_as_input_ops"]
#         op_idx_list_2 = self._var_op_deps[checkpoint_2]["var_as_output_ops"]
#         if op_idx_list_1 == [] or op_idx_list_2 == []:
#             return False
#         if min(op_idx_list_1) >= max(op_idx_list_2):
#             return False
#         return True

#     def _get_recompute_segments(self, checkpoints):
#         segments = []
#         start_idx = -1
#         while start_idx + 1 < len(checkpoints):
#             if start_idx == -1:
#                 end_idx = start_idx + 1
#                 ckpt_name = checkpoints[end_idx]
#                 if ckpt_name not in self._var_op_deps:
#                     start_idx += 1
#                     continue
#                 op_idx_list = self._var_op_deps[ckpt_name]["var_as_output_ops"]
#                 if op_idx_list:
#                     segments.append([0, max(op_idx_list) + 1])
#             else:
#                 end_idx = start_idx + 1
#                 ckpt_1 = checkpoints[start_idx]
#                 ckpt_2 = checkpoints[end_idx]
#                 if self._is_segment(ckpt_1, ckpt_2):
#                     min_idx = min(self._var_op_deps[ckpt_1]["var_as_input_ops"])
#                     max_idx = max(self._var_op_deps[ckpt_2]["var_as_output_ops"])
#                     segments.append([min_idx, max_idx + 1])

#             start_idx += 1

#         return segments

# def _insert_seed_ops(block, ops, dist_context):
#     op_types = [op.desc.type() for op in ops]
#     if "dropout" not in op_types:
#         return

#     op_idx = 0
#     while op_idx < len(ops):
#         cur_op = ops[op_idx]
#         if "grad" in cur_op.type:
#             break
#         if cur_op.type != "dropout":
#             op_idx += 1
#             continue
#         if cur_op.input("Seed") is not None and len(cur_op.input("Seed")):
#             op_idx += 1
#             continue

#         # insert seed op to guarantee that two dropout op have some outputs
#         op_unique_name = unique_name.generate("seed")
#         var_unique_name = unique_name.generate_with_ignorable_key(".".join(
#             [op_unique_name, 'tmp']))
#         out_var = block.create_var(
#             name=var_unique_name,
#             dtype='int32',
#             type=core.VarDesc.VarType.LOD_TENSOR,
#             persistable=False,
#             stop_gradient=False)

#         # set new out_var's distributed attribution
#         cur_op_dist_attr = dist_context.get_op_dist_attr_for_program(cur_op)
#         ref_dims_mapping = [-1]
#         ref_process_mesh = cur_op_dist_attr.process_mesh
#         _ = set_var_dist_attr(dist_context, out_var, ref_dims_mapping, ref_process_mesh)

#         seed = 0 if cur_op.attr("fix_seed") is False else int(cur_op.attr("seed"))
#         seed_op = block._insert_op_without_sync(
#             index=op_idx,
#             type="seed",
#             inputs={},
#             outputs={"Out": out_var},
#             attrs={
#                 "seed": seed,
#                 "force_cpu": True
#             })

#         # set new seed op's distributed attribution
#         seed_op_dist_attr = OperatorDistributedAttribute()
#         seed_op_dist_attr.process_mesh = ref_process_mesh
#         seed_op_dist_attr.impl_idx = cur_op_dist_attr.impl_idx
#         seed_op_dist_attr.set_output_dims_mapping(out_var.name, ref_dims_mapping)

#         # modify dropout op desc
#         ops.insert(op_idx, seed_op)
#         cur_op.desc.set_input("Seed", [var_unique_name])
#         cur_op.desc.remove_attr("fix_seed")
#         cur_op.desc.remove_attr("seed")
#         block._sync_with_cpp()
#         op_idx += 2

# def _get_stop_gradients(program, no_grad_set):
#     if no_grad_set is None:
#         no_grad_set = set()
#     else:
#         no_grad_set = _get_no_grad_set_name(no_grad_set)

#     no_grad_set_name = set()
#     for var in program.list_vars():
#         assert isinstance(var, Variable)
#         if var.stop_gradient:
#             no_grad_set_name.add(_append_grad_suffix_(var.name))
#     no_grad_set_name.update(list(map(_append_grad_suffix_, no_grad_set)))
#     return no_grad_set_name

# def _get_fwd_op_num(block):
#     fwd_op_num = 0
#     for op in block.ops:
#         if is_forward_op(op):
#             fwd_op_num += 1
#             continue
#         break
#     return fwd_op_num

# def _find_op_path_(block,
#                    targets,
#                    no_grad_set):

#     output_names = _get_output_names(block, targets)
#     relevant_op_flags = [True] * len(block.ops)

#     for i, op in reversed(list(enumerate(block.ops))):
#         if _some_in_set_(
#                 op.desc.output_arg_names(),
#                 output_names):
#             print("***op.desc.output_arg_names():", op.desc.output_arg_names())
#             print("***output_names:", output_names)
#             print("***op.desc.input_arg_names():", op.desc.input_arg_names())
#             print("***no_grad_set:", no_grad_set)
#             print("")
#             for name in op.desc.input_arg_names():
#                 if name not in no_grad_set:
#                     output_names.add(name)
#         else:
#             relevant_op_flags[i] = False

#     op_path = [
#         block.ops[i] for i in range(len(block.ops)) if relevant_op_flags[i]
#     ]

#     return op_path

# def _append_backward_ops_with_checkpoints_(
#         block, ops, target_block, no_grad_set, grad_to_var, checkpoints, dist_context):
#     recompute_state = RecomputeState(block, ops)
#     recompute_state._build_var_op_dependency()
#     checkpoints = recompute_state._sort_checkpoints(checkpoints)
#     segments = recompute_state._get_recompute_segments(checkpoints)
#     print("***segments:", segments)

# @register_pass("auto_parallel_recompute_forward")
# class RecomputeForwardPass(PassBase):
#     def __init__(self):
#         super(RecomputeForwardPass, self).__init__()
#         self.set_attr("dist_context", None)
#         self.set_attr("checkpoints", None)

#     def _check_self(self):
#         if self.get_attr("dist_context") is None:
#             return False
#         return True

#     def _check_conflict(self, other_pass):
#         return True

#     def _apply_single_impl(self, main_programs, startup_programs, context):
#         checkpoints = self.get_attr("checkpoints")
#         assert checkpoints is not None, "Checkpoints should be set in advance."

#         self._dist_context = self.get_attr("dist_context")
#         block = main_programs.global_block()
#         _insert_seed_ops(block, block.ops, self._dist_context)

# @register_pass("auto_parallel_recompute_backward")
# class RecomputeBackwardPass(PassBase):
#     def __init__(self):
#         super(RecomputeBackwardPass, self).__init__()
#         self.set_attr("checkpoints", None)
#         self.set_attr("dist_context", None)
#         self.set_attr("loss", None)
#         self.set_attr("parameter_list", None)
#         self.set_attr("no_grad_set", None)

#     def _check_self(self):
#         if self.get_attr("dist_context") is None:
#             return False
#         return True

#     def _check_conflict(self, other_pass):
#         return True

#     def _apply_single_impl(self, main_programs, startup_programs, context):
#         checkpoints = self.get_attr("checkpoints")
#         assert checkpoints is not None, "Checkpoints should be set in advance."

#         # find loss op
#         loss = self.get_attr("loss")
#         check_type(loss, 'loss', Variable, 'auto_parallel_recompute_backward')
#         if loss.op is None:
#             _find_loss_op_(loss)
#         loss.op._set_attr(core.op_proto_and_checker_maker.kOpRoleAttrName(),
#                         int(core.op_proto_and_checker_maker.OpRole.Forward) |
#                         int(core.op_proto_and_checker_maker.OpRole.Loss))

#         # get no grad var name
#         no_grad_set = self.get_attr("no_grad_set")
#         no_grad_set_name = _get_stop_gradients(main_programs, no_grad_set)

#         # get forward op number
#         main_block = main_programs.global_block()
#         fwd_op_mum = _get_fwd_op_num(main_block)

#         # append loss grad op
#         loss_op_desc = _create_loss_op_desc_(loss)
#         main_block.desc.append_op().copy_from(loss_op_desc)
#         print("***main_block.ops: ", len(main_block.ops), main_block.ops)
#         # get op_path which is related to loss
#         op_path = _find_op_path_(main_block, [loss], no_grad_set_name)
#         # get no_grad_vars which is not related to loss
#         no_grad_vars = _find_no_grad_vars(main_block, op_path, [loss], no_grad_set_name)
#         no_grad_set_name.update(list(map(_append_grad_suffix_, no_grad_vars)))

#         # append backward ops with checkpoints and dist_context
#         grad_to_var = dict()
#         dist_context = self.get_attr("dist_context")
#         _append_backward_ops_with_checkpoints_(main_block, op_path, main_block, no_grad_set_name, grad_to_var, checkpoints, dist_context)

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

import paddle
import logging
import paddle.compat as cpt

from .pass_base import PassBase, register_pass
from paddle.fluid import core, unique_name
from paddle.fluid.framework import Variable, Operator
from paddle.fluid.data_feeder import check_variable_and_dtype, check_type
from paddle.distributed.auto_parallel.process_mesh import ProcessMesh
from paddle.distributed.auto_parallel.utils import is_forward_op, is_backward_op
from paddle.distributed.auto_parallel.utils import get_loss_op, set_var_dist_attr
from paddle.distributed.auto_parallel.dist_attribute import OperatorDistributedAttribute
from paddle.fluid.backward import _append_grad_suffix_, _get_no_grad_set_name, _get_output_names, _some_in_set_
from paddle.fluid.backward import _create_loss_op_desc_, _find_no_grad_vars, _find_loss_op_, _rename_arg_, _find_op_path_


class RecomputeState:
    def __init__(self, block, ops, dist_context):
        self._block = block
        self._ops = ops
        self._dist_context = dist_context
        self._var_op_deps = {}
        self._grad_var_op_deps = {}

        self._sync_loss_op_index()

    @property
    def loss_op_index(self):
        return self._loss_op_index

    def _sync_loss_op_index(self):
        loss_op = get_loss_op(self._block)
        self._loss_op_index = _find_op_index(self._block, loss_op) + 1

    def _build_dependency(self):
        for i, op in enumerate(self._ops):
            for name in op.desc.input_arg_names():
                if name in self._var_op_deps:
                    self._var_op_deps[name]["var_as_input_ops"].extend([i])
                else:
                    self._var_op_deps[name] = {}
                    self._var_op_deps[name]["var_as_input_ops"] = [i]
                    self._var_op_deps[name]["var_as_output_ops"] = []

            for name in op.desc.output_arg_names():
                if name in self._var_op_deps:
                    self._var_op_deps[name]["var_as_output_ops"].extend([i])
                else:
                    self._var_op_deps[name] = {}
                    self._var_op_deps[name]["var_as_input_ops"] = []
                    self._var_op_deps[name]["var_as_output_ops"] = [i]

        self._block._sync_with_cpp()
        for i, op in enumerate(self._block.ops):
            if i > self.loss_op_index:
                for name in op.desc.input_arg_names():
                    if name in self._grad_var_op_deps:
                        self._grad_var_op_deps[name]["var_as_input_ops"].extend(
                            [i])
                    else:
                        self._grad_var_op_deps[name] = {}
                        self._grad_var_op_deps[name]["var_as_input_ops"] = [i]
                        self._grad_var_op_deps[name]["var_as_output_ops"] = []

                for name in op.desc.output_arg_names():
                    if name in self._grad_var_op_deps:
                        self._grad_var_op_deps[name][
                            "var_as_output_ops"].extend([i])
                    else:
                        self._grad_var_op_deps[name] = {}
                        self._grad_var_op_deps[name]["var_as_input_ops"] = []
                        self._grad_var_op_deps[name]["var_as_output_ops"] = [i]

    def _sort_checkpoints(self, checkpoints):
        sorted_checkpoints = []
        for ckpt_name in checkpoints:
            if ckpt_name not in self._var_op_deps:
                logging.info(
                    "Delete %s from checkpoints, because it is not used in paddle program."
                    % ckpt_name)
            elif self._var_op_deps[ckpt_name]["var_as_output_ops"] == []:
                sorted_checkpoints.append((ckpt_name, -1))
            else:
                sorted_checkpoints.append(
                    (ckpt_name,
                     max(self._var_op_deps[ckpt_name]["var_as_output_ops"])))
        sorted_checkpoints = sorted(sorted_checkpoints, key=lambda x: x[1])
        checkpoints = [ckpt[0] for ckpt in sorted_checkpoints]
        return checkpoints

    def _is_segment(self, checkpoint_1, checkpoint_2):
        if checkpoint_1 not in self._var_op_deps or checkpoint_2 not in self._var_op_deps:
            return False
        op_idx_list_1 = self._var_op_deps[checkpoint_1]["var_as_input_ops"]
        op_idx_list_2 = self._var_op_deps[checkpoint_2]["var_as_output_ops"]
        if op_idx_list_1 == [] or op_idx_list_2 == []:
            return False
        if min(op_idx_list_1) >= max(op_idx_list_2):
            return False
        return True

    def _get_recompute_segments(self, checkpoints):
        segments = []
        start_idx = -1
        while start_idx + 1 < len(checkpoints):
            if start_idx == -1:
                end_idx = start_idx + 1
                ckpt_name = checkpoints[end_idx]
                if ckpt_name not in self._var_op_deps:
                    start_idx += 1
                    continue
                op_idx_list = self._var_op_deps[ckpt_name]["var_as_output_ops"]
                if op_idx_list:
                    segments.append([0, max(op_idx_list) + 1])
            else:
                end_idx = start_idx + 1
                ckpt_1 = checkpoints[start_idx]
                ckpt_2 = checkpoints[end_idx]
                if self._is_segment(ckpt_1, ckpt_2):
                    min_idx = min(self._var_op_deps[ckpt_1]["var_as_input_ops"])
                    max_idx = max(self._var_op_deps[ckpt_2][
                        "var_as_output_ops"])
                    segments.append([min_idx, max_idx + 1])

            start_idx += 1

        return segments

    def _insert_seed_ops(self):  #  block, ops, dist_context
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

            # insert seed op to guarantee that two dropout op have some outputs
            op_unique_name = unique_name.generate("seed")
            var_unique_name = unique_name.generate_with_ignorable_key(".".join(
                [op_unique_name, 'tmp']))
            out_var = self._block.create_var(
                name=var_unique_name,
                dtype='int32',
                type=core.VarDesc.VarType.LOD_TENSOR,
                persistable=False,
                stop_gradient=False)

            # set new out_var's distributed attribution
            cur_op_dist_attr = self._dist_context.get_op_dist_attr_for_program(
                cur_op)
            ref_dims_mapping = [-1]
            ref_process_mesh = cur_op_dist_attr.process_mesh
            _ = set_var_dist_attr(self._dist_context, out_var, ref_dims_mapping,
                                  ref_process_mesh)

            seed = 0 if cur_op.attr("fix_seed") is False else int(
                cur_op.attr("seed"))
            seed_op = self._block._insert_op_without_sync(
                index=cur_op.idx,
                type="seed",
                inputs={},
                outputs={"Out": out_var},
                attrs={"seed": seed,
                       "force_cpu": True})

            # set new seed op's distributed attribution
            seed_op_dist_attr = OperatorDistributedAttribute()
            seed_op_dist_attr.process_mesh = ref_process_mesh
            seed_op_dist_attr.impl_idx = cur_op_dist_attr.impl_idx
            seed_op_dist_attr.set_output_dims_mapping(out_var.name,
                                                      ref_dims_mapping)

            # modify dropout op desc
            self._ops.insert(op_idx, seed_op)
            cur_op.desc.set_input("Seed", [var_unique_name])
            cur_op.desc.remove_attr("fix_seed")
            cur_op.desc.remove_attr("seed")
            self._block._sync_with_cpp()
            op_idx += 2

        self._sync_loss_op_index()

    def get_out_of_subgraph_vars(self, begin_op_idx, end_op_idx):
        var_name = []
        for i in range(begin_op_idx, end_op_idx, 1):
            for name in self._ops[i].desc.output_arg_names():
                if name in self._var_op_deps:
                    for idx in self._var_op_deps[name]["var_as_input_ops"]:
                        if idx >= end_op_idx:
                            var_name.append(name)
            for name in self._ops[i].desc.input_arg_names():
                if name in self._var_op_deps:
                    for idx in self._var_op_deps[name]["var_as_output_ops"]:
                        if idx < begin_op_idx:
                            var_name.append(name)
        return var_name

    def get_input_nodes(self):
        input_names = []
        for name in self._var_op_deps:
            if len(self._var_op_deps[name]["var_as_output_ops"]) == 0 and \
                    len(self._var_op_deps[name]["var_as_input_ops"]) > 0:
                if self._block.var(name).persistable:
                    continue
                input_names.append(name)
        for op in self._ops:
            if op.desc.type() == "read":
                input_names.extend(op.desc.output_arg_names())
        return input_names

    def get_reserved_vars(self):
        var_name = []
        for op in self._ops:
            if op.desc.type() == "seed":
                var_name.extend(op.desc.output_arg_names())
        return var_name


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


def _get_fwd_op_num(block):
    fwd_op_num = 0
    for op in block.ops:
        if is_forward_op(op):
            fwd_op_num += 1
            continue
        break
    return fwd_op_num


def _find_op_index(block, cur_op):
    for idx in range(block.desc.op_size()):
        if cur_op.desc == block.desc.op(idx):
            return idx
    return -1


def _add_descs_to_block(descs, block):
    if len(descs) == 0:
        return []
    result_descs = []
    op_role_attr_name = \
        core.op_proto_and_checker_maker.kOpRoleAttrName()
    backward = core.op_proto_and_checker_maker.OpRole.Backward
    for desc in descs:
        if isinstance(desc, Operator):
            desc = desc.desc
        if isinstance(desc, tuple):
            desc = desc[0]
        new_op_desc = block.desc.append_op()
        new_op_desc.copy_from(desc)
        new_op_desc._set_attr(op_role_attr_name, backward)
        if desc.has_attr('op_device'):
            new_op_desc._set_attr('op_device', desc.attr('op_device'))
        result_descs.append(new_op_desc)
    return result_descs


def _add_needed_descs_to_block(descs, block, main_block, in_memory_vars):
    if len(descs) == 0:
        return []
    result_descs = []
    op_role_attr_name = \
        core.op_proto_and_checker_maker.kOpRoleAttrName()
    backward = core.op_proto_and_checker_maker.OpRole.Backward
    for desc in descs:
        if isinstance(desc, Operator):
            desc = desc.desc
        if isinstance(desc, tuple):
            desc = desc[0]
        is_needed = False
        for name in desc.output_arg_names():
            if main_block.has_var(name) and main_block.var(name).persistable:
                continue
            if name not in in_memory_vars:
                is_needed = True
        if is_needed:
            new_op_desc = block.desc.append_op()
            new_op_desc.copy_from(desc)
            new_op_desc._set_attr(op_role_attr_name, backward)
            if desc.has_attr('op_device'):
                new_op_desc._set_attr('op_device', desc.attr('op_device'))
            result_descs.append(new_op_desc)
    return result_descs


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
        no_grad_vars = _find_no_grad_vars(main_block, op_path, [loss],
                                          no_grad_set_name)
        no_grad_set_name.update(list(map(_append_grad_suffix_, no_grad_vars)))

        # insert seed op in forward program
        dist_context = self.get_attr("dist_context")
        recompute_state = RecomputeState(main_block, op_path, dist_context)
        recompute_state._insert_seed_ops()
        recompute_state._build_dependency()
        checkpoints = recompute_state._sort_checkpoints(checkpoints)
        segments = recompute_state._get_recompute_segments(checkpoints)

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

        loss_op_idx = _find_op_index(main_block, loss_op)
        grad_start_op_idx = loss_op_idx + 1
        ops = main_block.ops

        var_name_dict = {}
        device_attr_name = core.op_proto_and_checker_maker.kOpDeviceAttrName()
        local_block = main_block.program._create_block()
        buffer_block = main_block.program._create_block()
        print("=====================================")
        print("***checkpoints:", checkpoints)
        if segments:
            for i, segment in enumerate(segments[::-1]):
                cur_ckpt = checkpoints[len(checkpoints) - i - 1]
                start_idx = grad_start_op_idx
                while cur_ckpt not in ops[start_idx].desc.input_arg_names():
                    start_idx += 1

                assert cur_ckpt in ops[start_idx].desc.input_arg_names()
                start_idx += 1

                fwd_ops = op_path[segment[0]:segment[1]]
                var_suffix = ".subprog_%d" % i

                for op in fwd_ops:
                    input_and_output_names = []
                    input_and_output_names.extend(op.desc.input_arg_names())
                    input_and_output_names.extend(op.desc.output_arg_names())
                    for name in input_and_output_names:
                        if main_block.var(
                                name).persistable or name in checkpoints:
                            continue
                        if name in vars_should_be_hold:
                            continue
                        if name not in var_name_dict:
                            var_name_dict[name] = name + var_suffix

                            # we should create the rename var in subprog, otherwise its VarType will be BOOL
                            ref_var = main_block.program.global_block().var(
                                name)
                            main_block.create_var(
                                name=var_name_dict[name],
                                shape=ref_var.shape,
                                dtype=ref_var.dtype,
                                type=ref_var.type,
                                persistable=ref_var.persistable,
                                stop_gradient=ref_var.stop_gradient)

                buffer_descs = _add_needed_descs_to_block(
                    fwd_ops, buffer_block, main_block, vars_in_memory)

                for key in var_name_dict:
                    _rename_arg_(buffer_descs, key, var_name_dict[key])

                # for _, op_desc in reversed(list(enumerate(buffer_descs))):
                #     cur_op = main_block.ops[start_idx]
                #     desc = main_block.desc._insert_op(cur_op.idx)
                #     desc.copy_from(op_desc)
                #     fwd_op = paddle.fluid.framework.Operator(main_block, desc)
                #     main_block.ops.insert(start_idx, fwd_op)

                # main_block._sync_with_cpp()
                # start_idx += len(buffer_descs)

                if len(checkpoints) - i - 2 >= 0:
                    pre_ckpt = checkpoints[len(checkpoints) - i - 2]
                    idx_list = recompute_state._grad_var_op_deps[pre_ckpt][
                        "var_as_input_ops"]
                    print("***pre_ckpt:", pre_ckpt)
                    pre_ckpt_idx = max(idx_list)
                    print("***idx_list:", idx_list)
                    print("***pre_ckpt_idx:", pre_ckpt_idx)
                    print("***start_idx:", start_idx)
                    while start_idx < pre_ckpt_idx:
                        grad_op_desc = main_block.ops[start_idx].desc
                        print(main_block.ops[start_idx])
                        for key in var_name_dict:
                            _rename_arg_([grad_op_desc], key,
                                         var_name_dict[key])
                        start_idx += 1

                    print("***start_idx:", start_idx)
                    print("***main_block.ops[start_idx]:",
                          main_block.ops[start_idx])
                    assert pre_ckpt in main_block.ops[
                        start_idx].desc.input_arg_names()
                    start_idx += 1
                else:
                    while start_idx < len(main_block.ops):
                        for key in var_name_dict:
                            grad_op_desc = main_block.ops[start_idx].desc
                            _rename_arg_([grad_op_desc], key,
                                         var_name_dict[key])
                        start_idx += 1

                grad_start_op_idx = start_idx
                main_block._sync_with_cpp()
                print("************************")
                print(main_block)
