# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from collections import defaultdict

import paddle

from .pass_base import PassBase, register_pass
from ..auto_parallel.utils import is_forward_op, is_optimize_op, OP_ROLE_KEY, OpRole, ring_id_to_process_group, print_program_with_dist_attr
from ..auto_parallel.operators.common import is_parameter_related, is_data_parallel_scale_op, is_data_parallel_reduce_op, ParallelMode
from ..auto_parallel.dist_attribute import OperatorDistributedAttribute


def _is_renamed_grad_var(var_name):
    return "@RENAME@block" in var_name


def _prefix_name(var_name):
    return var_name[:var_name.find("@RENAME")]


def _is_data_parallel_rename_scale_op(op):
    return is_data_parallel_scale_op(op) and \
        _is_renamed_grad_var(op.output_arg_names[0])


def _is_data_parallel_rename_reduce_op(op):
    return is_data_parallel_reduce_op(op) and \
        _is_renamed_grad_var(op.output_arg_names[0])


def _is_model_parallel_rename_reduce_op(op):
    return op.has_attr('use_model_parallel') and \
        op.attr('use_model_parallel') and \
            _is_renamed_grad_var(op.output_arg_names[0])


@register_pass("auto_parallel_allreduce_sum")
class AllReduceSumOptimizationPass(PassBase):

    def __init__(self):
        super(AllReduceSumOptimizationPass, self).__init__()
        self.set_attr("dist_context", None)

    def _check_self(self):
        if self.get_attr("dist_context") is None:
            return False
        # if self.get_attr()
        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, context):

        self.dist_ctx = self.get_attr('dist_context')
        self.dp_grad_name_to_attr = defaultdict(dict)
        self.mp_grad_name_to_attr = defaultdict(dict)
        self.main_block = main_program.global_block()

        self._build_grad_name_to_attr()
        self._remove_no_need_op()
        self._insert_allreduce_scale_op()
        # print_program_with_dist_attr(main_program, self.dist_ctx)

    def _build_grad_name_to_attr(self):
        for idx, op in enumerate(self.main_block.ops):
            if is_forward_op(op):
                continue

            if is_optimize_op(op):
                break

            if _is_data_parallel_rename_reduce_op(op):
                out_name = op.output_arg_names[0]
                grad_name = _prefix_name(out_name)

                assert op.has_attr(
                    "ring_id"), "ERROR: comm op [{}] has not ring id.".format(
                        str(op))
                group = ring_id_to_process_group(op.attr('ring_id'))

                if grad_name in self.dp_grad_name_to_attr:
                    # if self.dp_grad_name_to_attr[grad_name]['group'] != group:
                    #     self.dp_grad_name_to_attr.pop(grad_name)
                    #     continue
                    self.dp_grad_name_to_attr[grad_name]['group'] = group
                    self.dp_grad_name_to_attr[grad_name]['rename'].append(
                        out_name)
                    self.dp_grad_name_to_attr[grad_name]['remove_idx'].append(
                        idx)
                    continue

                info = {}
                info['group'] = group
                info['remove_idx'] = [idx]
                info['rename'] = [out_name]
                self.dp_grad_name_to_attr[grad_name] = info

            if _is_data_parallel_rename_scale_op(op):
                out_name = op.output_arg_names[0]
                grad_name = _prefix_name(out_name)

                # assert grad_name in self.dp_grad_name_to_attr, \
                #     "ERROR: gradients [{}] is scaled but not synchronized.".format(out_name)

                self.dp_grad_name_to_attr[grad_name]['remove_idx'].append(idx)

            if _is_model_parallel_rename_reduce_op(op):
                out_name = op.output_arg_names[0]
                grad_name = _prefix_name(out_name)

                assert op.has_attr(
                    "ring_id"), "ERROR: comm op [{}] has not ring id.".format(
                        str(op))
                group = ring_id_to_process_group(op.attr('ring_id'))

                if grad_name in self.mp_grad_name_to_attr:
                    self.mp_grad_name_to_attr[grad_name]['group'] = group
                    self.mp_grad_name_to_attr[grad_name]['rename'].append(
                        out_name)
                    self.mp_grad_name_to_attr[grad_name]['remove_idx'].append(
                        idx)
                    continue

                info = {}
                info['group'] = group
                info['remove_idx'] = [idx]
                info['rename'] = [out_name]
                self.mp_grad_name_to_attr[grad_name] = info

            if op.type == 'sum' and op.output_arg_names[
                    0] in self.dp_grad_name_to_attr:
                out_name = op.output_arg_names[0]
                rename_list = self.dp_grad_name_to_attr[out_name]['rename']
                if sorted(rename_list) != sorted(op.input_arg_names):
                    self.dp_grad_name_to_attr.pop(out_name)

            if op.type == 'sum' and op.output_arg_names[
                    0] in self.mp_grad_name_to_attr:
                out_name = op.output_arg_names[0]
                rename_list = self.mp_grad_name_to_attr[out_name]['rename']
                if sorted(rename_list) != sorted(op.input_arg_names):
                    self.mp_grad_name_to_attr.pop(out_name)

    def _remove_no_need_op(self):
        remove_op_idx = []
        for item in self.dp_grad_name_to_attr.values():
            remove_op_idx.extend(item['remove_idx'])

        for item in self.mp_grad_name_to_attr.values():
            remove_op_idx.extend(item['remove_idx'])
            for idx in item['remove_idx']:
                in_name = self.main_block.ops[idx].input_arg_names[0]
                out_name = self.main_block.ops[idx].output_arg_names[0]
                pre_op = self.main_block.vars[in_name].op
                pre_op.desc._rename_output(in_name, out_name)

        for idx in sorted(remove_op_idx, reverse=True):
            self.main_block._remove_op(idx, sync=False)
        self.main_block._sync_with_cpp()

    def _insert_allreduce_scale_op(self):
        for idx, op in reversed(list(enumerate(self.main_block.ops))):
            if is_forward_op(op):
                break
            if is_optimize_op(op):
                continue

            if op.type == 'sum' and op.output_arg_names[
                    0] in self.dp_grad_name_to_attr:

                out_name = op.output_arg_names[0]
                info = self.dp_grad_name_to_attr[out_name]
                group = info['group']
                in_var = self.main_block.vars[out_name]
                allreduce_op = self.main_block._insert_op_without_sync(
                    idx + 1,
                    type="c_allreduce_sum",
                    inputs={'X': in_var},
                    outputs={'Out': in_var},
                    attrs={
                        'ring_id': group.id,
                        'use_calc_stream': True,
                        OP_ROLE_KEY: OpRole.Backward
                    })
                allreduce_op._set_attr('op_namescope',
                                       str('/') + ParallelMode.DataParallel)

                sum_op_dist_attr = self.dist_ctx.get_op_dist_attr_for_program(
                    op)
                process_mesh = sum_op_dist_attr.process_mesh
                dims_mapping = sum_op_dist_attr.get_output_dims_mapping(
                    in_var.name)
                new_op_dist_attr = OperatorDistributedAttribute()
                new_op_dist_attr.process_mesh = process_mesh
                new_op_dist_attr.set_input_dims_mapping(in_var.name,
                                                        dims_mapping)
                new_op_dist_attr.set_output_dims_mapping(
                    in_var.name, dims_mapping)
                self.dist_ctx.set_op_dist_attr_for_program(
                    allreduce_op, new_op_dist_attr)

                if self.dist_ctx.gradient_scale:
                    # if 'scale' in info and info['scale'] != 1.0:
                    # scale_value = info['scale']
                    scale_op = self.main_block._insert_op_without_sync(
                        idx + 2,
                        type="scale",
                        inputs={'X': in_var},
                        outputs={'Out': in_var},
                        attrs={
                            'scale': 1.0 / len(group.ranks),
                            OP_ROLE_KEY: OpRole.Backward
                        })
                    scale_op._set_attr('op_namescope',
                                       str('/') + ParallelMode.DataParallel)

                    new_op_dist_attr = OperatorDistributedAttribute()
                    new_op_dist_attr.process_mesh = process_mesh
                    new_op_dist_attr.set_input_dims_mapping(
                        in_var.name, dims_mapping)
                    new_op_dist_attr.set_output_dims_mapping(
                        in_var.name, dims_mapping)
                    self.dist_ctx.set_op_dist_attr_for_program(
                        scale_op, new_op_dist_attr)

            if op.type == 'sum' and op.output_arg_names[
                    0] in self.mp_grad_name_to_attr:
                out_name = op.output_arg_names[0]
                info = self.mp_grad_name_to_attr[out_name]
                group = info['group']
                in_var = self.main_block.vars[out_name]
                allreduce_op = self.main_block._insert_op_without_sync(
                    idx + 1,
                    type="c_allreduce_sum",
                    inputs={'X': in_var},
                    outputs={'Out': in_var},
                    attrs={
                        'ring_id': group.id,
                        'use_calc_stream': True,
                        'use_model_parallel': True,
                        OP_ROLE_KEY: OpRole.Backward
                    })
                sum_op_dist_attr = self.dist_ctx.get_op_dist_attr_for_program(
                    op)
                process_mesh = sum_op_dist_attr.process_mesh
                dims_mapping = sum_op_dist_attr.get_output_dims_mapping(
                    in_var.name)
                new_op_dist_attr = OperatorDistributedAttribute()
                new_op_dist_attr.process_mesh = process_mesh
                new_op_dist_attr.set_input_dims_mapping(in_var.name,
                                                        dims_mapping)
                new_op_dist_attr.set_output_dims_mapping(
                    in_var.name, dims_mapping)
                self.dist_ctx.set_op_dist_attr_for_program(
                    allreduce_op, new_op_dist_attr)

        self.main_block._sync_with_cpp()
