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

import os
import copy
from .framework import _apply_pass


def get_data_vars(program):
    data_vars = []
    for var_name, var in program.global_block().vars.items():
        if var.is_data:
            data_vars.append(var_name)
    return data_vars


def apply_build_strategy(main_program, startup_program, build_strategy,
                         pass_attrs):
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
        update_attr(attrs, attr_types, "mem_opt_skip_vars",
                    get_data_vars(main_program), "list[str]")
        _apply_pass(main_program, startup_program, name, attrs, attr_types)

    use_cuda = pass_attrs.get("use_cuda", False)
    build_strategy = build_strategy._copy()
    if build_strategy.sync_batch_norm:
        apply_pass("sync_batch_norm_pass")
        build_strategy.sync_batch_norm = False
    if build_strategy.fuse_relu_depthwise_conv and use_cuda:
        apply_pass("fuse_relu_depthwise_conv_pass")
        build_strategy.fuse_relu_depthwise_conv = False
    if build_strategy.fuse_bn_act_ops and use_cuda:
        apply_pass("fuse_bn_act_pass")
        build_strategy.fuse_bn_act_ops = False
    if build_strategy.fuse_bn_add_act_ops and use_cuda:
        apply_pass("fuse_bn_add_act_pass")
        build_strategy.fuse_bn_add_act_ops = False
    if build_strategy.enable_auto_fusion and use_cuda:
        apply_pass("fusion_group_pass")
        build_strategy.enable_auto_fusion = False
    if build_strategy.fuse_elewise_add_act_ops:
        apply_pass("fuse_elewise_add_act_pass")
        build_strategy.fuse_elewise_add_act_ops = False
    if build_strategy.fuse_all_optimizer_ops:
        apply_pass("fuse_adam_op_pass")
        apply_pass("fuse_sgd_op_pass")
        apply_pass("fuse_momentum_op_pass")
        build_strategy.fuse_all_optimizer_ops = False
    # TODO(zjl): support fuse all reduce ops
    if build_strategy.cache_runtime_context:
        apply_pass("runtime_context_cache_pass")
        build_strategy.cache_runtime_context = False
    if build_strategy.enable_addto and use_cuda:
        # NOTE: how to get fetch vars to skip memory optimization?  
        apply_pass("inplace_addto_op_pass")
        build_strategy.enable_addto = False
    if build_strategy.enable_inplace:
        apply_pass("buffer_shared_inplace_pass")
        build_strategy.enable_inplace = False
    build_strategy._clear_finalized()
    return build_strategy
