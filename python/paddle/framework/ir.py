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

from ..base.framework import _apply_pass
from . import core


def get_data_vars(program):
    data_vars = []
    for var_name, var in program.global_block().vars.items():
        if var.is_data:
            data_vars.append(var_name)
    return data_vars


def _update_grad_persistable(main_program):
    grad_merge_attr_name = "grad_merge_cond_name"
    op_role_var_attr_name = core.op_proto_and_checker_maker.kOpRoleVarAttrName()
    has_grad_merge = False
    has_persistable_grad_var = False
    grad_vars = []
    for block_id in range(main_program.num_blocks):
        block = main_program.block(block_id)
        for op in block.ops:
            if grad_merge_attr_name in op.attr_names:
                has_grad_merge = True

            if op_role_var_attr_name not in op.attr_names:
                continue

            p_g = op.attr(op_role_var_attr_name)
            for g in p_g[1::2]:
                g_var = block._find_var_recursive(g)
                if g_var is None:
                    continue
                grad_vars.append(g_var)
                if g_var.persistable:
                    has_persistable_grad_var = True

    if has_grad_merge and has_persistable_grad_var:
        for g_var in grad_vars:
            g_var.persistable = True


def apply_build_strategy(
    main_program, startup_program, build_strategy, pass_attrs
):
    def update_attr(attrs, attr_types, name, value, typ=None):
        if name not in attrs:
            attrs[name] = value
        if typ:
            attr_types[name] = typ

    def apply_pass(name):
        attrs = dict(pass_attrs)
        attr_types = {}
        update_attr(attrs, attr_types, "nranks", 1, "size_t")
        update_attr(attrs, attr_types, "use_cuda", False, "bool")
        # TODO(zjl): how to skip fetch variables ?
        update_attr(
            attrs,
            attr_types,
            "mem_opt_skip_vars",
            get_data_vars(main_program),
            "list[str]",
        )
        _apply_pass(main_program, startup_program, name, attrs, attr_types)

    _update_grad_persistable(main_program)
    use_cuda = pass_attrs.get("use_cuda", False)
    build_strategy = build_strategy._copy()
    if build_strategy.sync_batch_norm:
        apply_pass("sync_batch_norm_pass")
        build_strategy.sync_batch_norm = False
    if build_strategy.fuse_relu_depthwise_conv and use_cuda:
        apply_pass("fuse_relu_depthwise_conv_pass")
        build_strategy.fuse_relu_depthwise_conv = False
    if build_strategy.fuse_resunit:
        apply_pass("fuse_resunit_pass")
        build_strategy.fuse_resunit = False
    if build_strategy.fuse_bn_act_ops and use_cuda:
        apply_pass("fuse_bn_act_pass")
        build_strategy.fuse_bn_act_ops = False
    if build_strategy.fuse_bn_add_act_ops and use_cuda:
        apply_pass("fuse_bn_add_act_pass")
        build_strategy.fuse_bn_add_act_ops = False
    if build_strategy.enable_auto_fusion and use_cuda:
        apply_pass("fusion_group_pass")
        build_strategy.enable_auto_fusion = False
    if build_strategy.fuse_gemm_epilogue:
        apply_pass("fuse_gemm_epilogue_pass")
        build_strategy.fuse_gemm_epilogue = False
    if build_strategy.fuse_dot_product_attention:
        apply_pass("fuse_dot_product_attention_pass")
        build_strategy.fuse_dot_product_attention = False
    if build_strategy.fuse_elewise_add_act_ops:
        apply_pass("fuse_elewise_add_act_pass")
        build_strategy.fuse_elewise_add_act_ops = False
    if build_strategy.fuse_all_optimizer_ops:
        apply_pass(
            [
                "coalesce_grad_tensor_pass",
                "fuse_adam_op_pass",
                "fuse_sgd_op_pass",
                "fuse_momentum_op_pass",
            ]
        )
        build_strategy.fuse_all_optimizer_ops = False
    # TODO(zjl): support fuse all reduce ops
    if build_strategy.cache_runtime_context:
        apply_pass("runtime_context_cache_pass")
        build_strategy.cache_runtime_context = False
    build_strategy._clear_finalized()
    return build_strategy
