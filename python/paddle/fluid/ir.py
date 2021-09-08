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
import inspect
import paddle
from . import core
from .framework import _apply_pass, OpProtoHolder
from . import unique_name

try:
    from .proto import pass_desc_pb2
except ModuleNotFoundError:
    import sys
    from .proto import framework_pb2
    sys.path.append(framework_pb2.__file__.rsplit('/', 1)[0])
    from .proto import pass_desc_pb2


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

    _update_grad_persistable(main_program)
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
        apply_pass([
            "coalesce_grad_tensor_pass",
            "fuse_adam_op_pass",
            "fuse_sgd_op_pass",
            "fuse_momentum_op_pass",
        ])
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


class RegisterPassHelper(object):
    def __init__(self):
        self._pass_type = ""
        self._pass_pairs = None
        self._input_specs = dict()

    def _get_args_from_func(self, func):
        args = list()
        arg_specs = inspect.getfullargspec(func)
        for arg_name in arg_specs.args:
            input_spec = self._input_specs.get(arg_name)
            if isinstance(input_spec, paddle.static.InputSpec):
                args.append(
                    paddle.static.data(arg_name, input_spec.shape,
                                       input_spec.dtype))
            elif isinstance(input_spec, paddle.ParamAttr):
                args.append(paddle.ParamAttr(arg_name))
            else:
                args.append(paddle.static.data(arg_name, [-1]))
        return args

    def _func_to_program_desc(self, func, program_desc, is_replace=False):
        vars = list()
        program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(program, startup_program):
            args = self._get_args_from_func(func)
            for arg in args:
                vars.append(arg.name)
            outs = func(*args)
            if not isinstance(outs, (list, tuple)):
                outs = [outs]
            for out in outs:
                if isinstance(out, PassDesc.OpHelper):
                    if out == PassDesc.OP_ANY:
                        raise NotImplementedError(
                            "Specific OP 'PassDesc.OP_ANY' is not implemented.")
                    else:
                        for out in out.Outputs().values():
                            vars.extend(out)
                elif isinstance(out, paddle.fluid.framework.Variable):
                    vars.append(out.name)
        program_desc.ParseFromString(program.desc.serialize_to_string())
        if is_replace:
            attrs = list()
            for op in program.current_block().ops:
                if not isinstance(op, PassDesc.OpHelper):
                    continue
                attrs.extend(op._attrs.values())
            return vars, attrs
        return vars

    def SetPassPairs(self, pass_pairs):
        self._pass_pairs = pass_pairs

    def SetInputSpecs(self, input_specs):
        if isinstance(input_specs, dict):
            self._input_specs = input_specs

    def SerializeMultiPassDesc(self):
        switch_static_mode = paddle.in_dynamic_mode()
        if switch_static_mode:
            paddle.enable_static()
        multi_pass_desc = pass_desc_pb2.MultiPassDesc()
        multi_pass_desc.pass_type = self._pass_type
        for (pattern, replace) in self._pass_pairs:
            pass_desc = multi_pass_desc.pass_descs.add()
            pattern_vars = self._func_to_program_desc(pattern,
                                                      pass_desc.pattern)
            replace_vars, attrs = self._func_to_program_desc(
                replace, pass_desc.replace, is_replace=True)
            for (pattern_var, replace_var) in zip(pattern_vars, replace_vars):
                var_map = pass_desc.var_maps.add()
                var_map.pattern_var = pattern_var
                var_map.replace_var = replace_var
            pattern_op_idxs = dict()
            for (idx, op) in enumerate(pass_desc.pattern.blocks[0].ops):
                op_idxs = pattern_op_idxs.get(op.type)
                if op_idxs:
                    op_idxs.append(idx)
                else:
                    pattern_op_idxs[op.type] = [idx]
            for attr in attrs:
                attr_map = pass_desc.attr_maps.add()
                attr_map.pattern_op_idx = pattern_op_idxs[
                    attr._pattern_op_type][attr._pattern_op_idx]
                attr_map.replace_op_idx = attr._replace_op_idx
                attr_map.pattern_name = attr._pattern_name
                attr_map.replace_name = attr._replace_name
        if switch_static_mode:
            paddle.disable_static()
        return multi_pass_desc.SerializeToString()


class PassDesc(object):
    class AttrHelper(object):
        def __init__(self, name, replace_op_idx):
            self._pattern_op_type = None
            self._pattern_op_idx = -1
            self._replace_op_idx = replace_op_idx
            self._pattern_name = name
            self._replace_name = name

        def ReusePattern(self, op, index=0, name=None):
            if name:
                self._pattern_name = name
            self._pattern_op_type = op
            self._pattern_op_idx = index

    class OpHelper(object):
        def __getattr__(self, type):
            op = PassDesc.OpHelper()
            op.Init(type)
            return op

        def __call__(self, *args, **kwargs):
            for (in_name, in_args) in kwargs.items():
                in_arg_names = list()
                if isinstance(in_args, (list, tuple)):
                    if len(in_args) == 0:
                        raise ValueError("Argument {} of is not allowd empty.".
                                         format(in_name, self._type))
                else:
                    in_args = [in_args]
                for in_arg in in_args:
                    if isinstance(in_arg, PassDesc.OpHelper):
                        in_arg_names.extend(in_arg.Output())
                    else:
                        in_arg_names.append(in_arg.name)
                self._op_desc.set_input(in_name, in_arg_names)
            return self

        def Init(self, type):
            block = paddle.static.default_main_program().current_block()
            self._type = type
            self._attrs = dict()
            self._op_idx = len(block.ops)
            self._op_desc = block.desc.append_op()
            self._op_desc.set_type(type)
            self._op_proto = OpProtoHolder.instance().get_op_proto(type)
            block.ops.append(self)

        def Attr(self, name):
            attr = self._attrs.get(name)
            if attr:
                return attr
            attr = PassDesc.AttrHelper(name, self._op_idx)
            self._attrs[name] = attr
            return attr

        def SetAttr(self, name, value):
            self._op_desc._set_attr(name, value)

        def Output(self, name=None):
            if name:
                return self.Outputs()[name]
            return list(self.Outputs().values())[0]

        def Outputs(self):
            outputs = self._op_desc.outputs()
            if len(outputs) > 0:
                return outputs
            block = paddle.static.default_main_program().current_block()
            for output_proto in self._op_proto.outputs:
                name = unique_name.generate(self._type)
                output = block.create_var(name=name)
                self._op_desc.set_output(output_proto.name, [name])
            return self._op_desc.outputs()

    OP = OpHelper()
    OP_ANY = OpHelper()


def RegisterPass(pass_name, input_specs=None):
    """register pass
    """

    def _is_pass_pair(check_pair):
        if isinstance(check_pair, (list, tuple)):
            if len(check_pair) == 2:
                if all(map(inspect.isfunction, check_pair)):
                    return True
        return False

    def _register_pass_warpper(func):
        register_func_sign = inspect.signature(func)
        if len(register_func_sign.parameters) == 1:
            raise NotImplementedError(
                "register pass with graph is not implemented.")
        elif len(register_func_sign.parameters) == 0:
            pass_pairs = func()
            if _is_pass_pair(pass_pairs):
                pass_pairs = [pass_pairs]
            elif not all(map(_is_pass_pair, pass_pairs)):
                raise ValueError("Error pass functions.")
            register_helper = RegisterPassHelper()
            register_helper.SetPassPairs(pass_pairs)
            register_helper.SetInputSpecs(input_specs)
        else:
            raise ValueError("Error pass functions.")
        return func

    return _register_pass_warpper
