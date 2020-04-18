#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import collections
import six

from paddle.fluid import core
from paddle.fluid.framework import Block

from paddle.fluid.incubate.fleet.parameter_server.ir.public import _get_optimize_ops
from paddle.fluid.incubate.fleet.parameter_server.ir.public import _orig_varname
from paddle.fluid.incubate.fleet.parameter_server.ir.public import _get_varname_parts
from paddle.fluid.incubate.fleet.parameter_server.ir.public import DistributedMode

LOOKUP_TABLE_TYPE = "lookup_table"
LOOKUP_TABLE_GRAD_TYPE = "lookup_table_grad"
OP_NAME_SCOPE = "op_namescope"
CLIP_OP_NAME_SCOPE = "@CLIP"
OP_ROLE_VAR_ATTR_NAME = core.op_proto_and_checker_maker.kOpRoleVarAttrName()
RPC_OP_ROLE_ATTR_NAME = op_role_attr_name = core.op_proto_and_checker_maker.kOpRoleAttrName(
)
OPT_OP_ROLE_ATTR_VALUE = core.op_proto_and_checker_maker.OpRole.Optimize
RPC_OP_ROLE_ATTR_VALUE = core.op_proto_and_checker_maker.OpRole.RPC
DIST_OP_ROLE_ATTR_VALUE = core.op_proto_and_checker_maker.OpRole.Dist
LR_SCHED_OP_ROLE_ATTR_VALUE = core.op_proto_and_checker_maker.OpRole.LRSched


def _is_optimizer_op(op):
    if "Param" in op.input_names and \
            "LearningRate" in op.input_names:
        return True
    return False


def _same_or_split_var(p_name, var_name):
    return p_name == var_name or p_name.startswith(var_name + ".block")


def _get_pserver_grad_param_var(var, var_dict):
    """
    Return pserver side grad/param variable, return None
    if the variable is not grad/param, e.g.

        a@GRAD -> a@GRAD.block0
        a@GRAD -> a@GRAD (a is not split)
        fc_0.w_0 -> fc_0.w_0.block_0
        fc_0.w_0 -> fc_0.w_0 (weight is not split)
        _generated_var_123 -> None
    """
    grad_block = None
    for _, g in six.iteritems(var_dict):
        if _orig_varname(g.name) == _orig_varname(var.name):
            # skip per trainer vars
            if g.name.find(".trainer_") == -1:
                # only param or grads have split blocks
                if _orig_varname(g.name) in self.grad_name_to_param_name or \
                        _orig_varname(g.name) in self.param_name_to_grad_name:
                    grad_block = g
                    break
    return grad_block


def _get_optimizer_input_shape(op_type, varkey, orig_shape, param_shape):
    """
    Returns the shape for optimizer inputs that need to be reshaped when
    Param and Grad is split to multiple servers.
    """
    # HACK(typhoonzero): Should use functions of corresponding optimizer in
    # optimizer.py to get the shape, do not  bind this in the transpiler.
    if op_type == "adam":
        if varkey in ["Moment1", "Moment2"]:
            return param_shape
    elif op_type == "adagrad":
        if varkey == "Moment":
            return param_shape
    elif op_type == "adamax":
        if varkey in ["Moment", "InfNorm"]:
            return param_shape
    elif op_type in ["momentum", "lars_momentum"]:
        if varkey == "Velocity":
            return param_shape
    elif op_type == "rmsprop":
        if varkey in ["Moment", "MeanSquare"]:
            return param_shape
    elif op_type == "decayed_adagrad":
        if varkey == "Moment":
            return param_shape
    elif op_type == "ftrl":
        if varkey in ["SquaredAccumulator", "LinearAccumulator"]:
            return param_shape
    elif op_type == "sgd":
        pass
    else:
        raise ValueError(
            "Not supported optimizer for distributed training: %s" % op_type)
    return orig_shape


def _append_pserver_non_opt_ops(optimize_block, opt_op, origin_program):
    program = optimize_block.program
    # Append the ops for parameters that do not need to be optimized/updated
    inputs = _get_input_map_from_op(origin_program.global_block().vars, opt_op)
    for key, varlist in six.iteritems(inputs):
        if not isinstance(varlist, list):
            varlist = [varlist]
        for i in range(len(varlist)):
            var = varlist[i]
            # for ops like clipping and weight decay, get the split var (xxx.block0)
            # for inputs/outputs
            grad_block = _get_pserver_grad_param_var(
                var, program.global_block().vars)
            if grad_block:
                varlist[i] = grad_block
            elif var.name not in program.global_block().vars:
                tmpvar = program.global_block()._clone_variable(var)
                varlist[i] = tmpvar
            else:
                varlist[i] = program.global_block().vars[var.name]
        inputs[key] = varlist

    outputs = _get_output_map_from_op(origin_program.global_block().vars,
                                      opt_op)
    for key, varlist in six.iteritems(outputs):
        if not isinstance(varlist, list):
            varlist = [varlist]
        for i in range(len(varlist)):
            var = varlist[i]
            grad_block = _get_pserver_grad_param_var(
                var, program.global_block().vars)
            if grad_block:
                varlist[i] = grad_block
            elif var.name not in program.global_block().vars:
                tmpvar = program.global_block()._clone_variable(var)
                varlist[i] = tmpvar
            else:
                varlist[i] = program.global_block().vars[var.name]
        outputs[key] = varlist

    return optimize_block.append_op(
        type=opt_op.type,
        inputs=inputs,
        outputs=outputs,
        attrs=opt_op.all_attrs())


def _append_pserver_grad_merge_ops(optimize_block, grad_varname_for_block,
                                   endpoint, grad_to_block_id, origin_program,
                                   trainers, is_sync):
    program = optimize_block.program
    pserver_block = program.global_block()
    grad_block = None
    for g in self.param_grad_ep_mapping[endpoint]["grads"]:
        if _orig_varname(g.name) == \
                _orig_varname(grad_varname_for_block):
            grad_block = g
            break
    if not grad_block:
        # do not append this op if current endpoint
        # is not dealing with this grad block
        return None

    orig_varname, block_name, trainer_name = _get_varname_parts(grad_block.name)
    if block_name:
        merged_var_name = '.'.join([orig_varname, block_name])
    else:
        merged_var_name = orig_varname

    merged_var = pserver_block.vars[merged_var_name]
    grad_to_block_id.append(merged_var.name + ":" + str(optimize_block.idx))
    if is_sync and trainers > 1:
        vars2merge = []
        for i in range(trainers):
            per_trainer_name = "%s.trainer_%d" % \
                               (merged_var_name, i)
            vars2merge.append(pserver_block.vars[per_trainer_name])
        optimize_block.append_op(
            type="sum",
            inputs={"X": vars2merge},
            outputs={"Out": merged_var},
            attrs={"use_mkldnn": False})
        optimize_block.append_op(
            type="scale",
            inputs={"X": merged_var},
            outputs={"Out": merged_var},
            attrs={"scale": 1.0 / float(trainers)})
    return merged_var


def _clone_lr_op(program, origin_program, block, op):
    inputs = _get_input_map_from_op(origin_program.global_block().vars, op)
    for key, varlist in six.iteritems(inputs):
        if not isinstance(varlist, list):
            varlist = [varlist]
        for var in varlist:
            if var not in program.global_block().vars:
                block._clone_variable(var)

    outputs = _get_output_map_from_op(origin_program.global_block().vars, op)
    for key, varlist in six.iteritems(outputs):
        if not isinstance(varlist, list):
            varlist = [varlist]
        for var in varlist:
            if var not in program.global_block().vars:
                block._clone_variable(var)

    return block.append_op(
        type=op.type, inputs=inputs, outputs=outputs, attrs=op.all_attrs())


def _append_pserver_ops(optimize_block, opt_op, endpoint, grad_to_block_id,
                        origin_program, merged_var, sparse_grad_to_param):
    program = optimize_block.program
    pserver_block = program.global_block()
    new_inputs = collections.OrderedDict()

    def _get_param_block(opt_op):
        # param is already created on global program
        param_block = None
        for p in self.param_grad_ep_mapping[endpoint]["params"]:
            if _same_or_split_var(p.name, opt_op.input("Param")[0]):
                param_block = p
                break
        return param_block

    for key in opt_op.input_names:
        if key == "Grad":
            # Note!! This is for l2decay on sparse gradient, because it will create a new tensor for
            # decayed gradient but not inplace modify the origin one
            origin_grad_name = opt_op.input(key)[0]
            if core.kNewGradSuffix(
            ) in origin_grad_name and pserver_block.has_var(origin_grad_name):
                new_grad = pserver_block.var(origin_grad_name)
                new_inputs[key] = new_grad
            else:
                new_inputs[key] = merged_var
        elif key == "Param":
            param_block = _get_param_block(opt_op)
            if not param_block:
                return
            tmpvar = pserver_block.create_var(
                name=param_block.name,
                persistable=True,
                dtype=param_block.dtype,
                shape=param_block.shape)
            new_inputs[key] = tmpvar
        elif key == "LearningRate":
            # learning rate variable has already be created by non-optimize op,
            # don't create it once again.
            lr_varname = opt_op.input(key)[0]
            if lr_varname in pserver_block.vars:
                new_inputs[key] = pserver_block.vars[opt_op.input(key)[0]]
            else:
                origin_var = origin_program.global_block().vars[lr_varname]
                tmpvar = pserver_block.create_var(
                    name=origin_var.name,
                    persistable=origin_var.persistable,
                    dtype=origin_var.dtype,
                    shape=origin_var.shape)
                new_inputs[key] = tmpvar

    for key in opt_op.input_names:
        new_shape = None
        if key in [
                "Param", "Grad", "LearningRate", "Beta1Tensor", "Beta2Tensor"
        ]:
            continue
        var = origin_program.global_block().vars[opt_op.input(key)[0]]
        param_var = new_inputs["Param"]
        # update accumulator variable shape
        new_shape = _get_optimizer_input_shape(opt_op.type, key, var.shape,
                                               param_var.shape)
        tmpvar = pserver_block.create_var(
            name=var.name,
            persistable=var.persistable,
            dtype=var.dtype,
            shape=new_shape)
        new_inputs[key] = tmpvar

    # change output's ParamOut variable
    outputs = _get_output_map_from_op(origin_program.global_block().vars,
                                      opt_op)
    outputs["ParamOut"] = new_inputs["Param"]
    optimize_block.append_op(
        type=opt_op.type,
        inputs=new_inputs,
        outputs=outputs,
        attrs=opt_op.all_attrs())

    # record sparse grad to param name
    if new_inputs["Grad"].type == core.VarDesc.VarType.SELECTED_ROWS:
        sparse_grad_to_param.append(
            str(new_inputs["Grad"].name) + ":" + str(new_inputs["Param"].name))


def _get_input_map_from_op(varmap, op):
    """Returns a dict from op input name to the vars in varmap."""
    iomap = collections.OrderedDict()
    for key in op.input_names:
        vars = []
        for varname in op.input(key):
            vars.append(varmap[varname])
        if len(vars) == 1:
            iomap[key] = vars[0]
        else:
            iomap[key] = vars
    return iomap


def _get_output_map_from_op(varmap, op):
    """Returns a dict from op output name to the vars in varmap."""
    iomap = collections.OrderedDict()
    for key in op.output_names:
        vars = []
        for varname in op.output(key):
            vars.append(varmap[varname])
        if len(vars) == 1:
            iomap[key] = vars[0]
        else:
            iomap[key] = vars
    return iomap


def get_op_by_type(block, op_type):
    for op in block:
        if op.type == op_type:
            return op
    raise ValueError("add_listen_and_serv_pass must at first")


def add_listen_and_serv_pass(program, pserver_id, endpoint, trainers,
                             distributed_mode):
    attrs = {
        "grad_to_block_id": None,
        "sparse_grad_to_param": None,
        "lr_decay_block_id": None,
        "dense_optimize_blocks": None,
        "sparse_optimize_blocks": None,

        # runtime attribute
        "endpoint": endpoint,
        "pserver_id": pserver_id,
        "Fanin": trainers,
        "distributed_mode": distributed_mode,
        "rpc_get_thread_num": -1,
        "rpc_send_thread_num": -1,
        "rpc_prefetch_thread_num": -1
    }

    # step5 append the listen_and_serv op
    program.global_block().append_op(
        type="listen_and_serv", inputs={'X': []}, outputs={}, attrs=attrs)

    return program


def add_rpc_global_flags_pass(program, config):
    server_runtime = config.get_server_runtime_config()
    send_threads = server_runtime._rpc_send_thread_num,
    get_threads = server_runtime._rpc_get_thread_num,
    pull_threads = server_runtime._rpc_prefetch_thread_num

    op = get_op_by_type(program.global_block(), "listen_and_serv")

    if get_threads <= 1 or send_threads <= 1 or pull_threads <= 1:
        raise ValueError(
            "error arguments in get_threads/send_threads/pull_threads")

    op._set_attr("rpc_get_thread_num", get_threads)
    op._set_attr("rpc_send_thread_num", send_threads)
    op._set_attr("rpc_prefetch_thread_num", pull_threads)

    return program


def _clone_var(self, block, var, persistable=True):
    return block.create_var(
        name=var.name,
        shape=var.shape,
        dtype=var.dtype,
        type=var.type,
        lod_level=var.lod_level,
        persistable=persistable)


def add_recv_inputs_pass(program, config):
    mode = config.get_distributed_mode
    trainers = config.get_trainers()

    for v in self.param_grad_ep_mapping[endpoint]["params"]:
        _clone_var(pserver_program.global_block(), v)
    for v in self.param_grad_ep_mapping[endpoint]["grads"]:
        # create vars for each trainer in global scope, so
        # we don't need to create them when grad arrives.
        # change client side var name to origin name by
        # removing ".trainer_%d" suffix
        suff_idx = v.name.find(".trainer_")
        if suff_idx >= 0:
            orig_var_name = v.name[:suff_idx]
        else:
            orig_var_name = v.name
        # NOTE: single_trainer_var must be created for multi-trainer
        # case to merge grads from multiple trainers
        input_var = \
            program.global_block().create_var(
                name=orig_var_name,
                persistable=True,
                type=v.type,
                dtype=v.dtype,
                shape=v.shape)
        if mode == DistributedMode.SYNC and trainers > 1:
            for trainer_id in range(trainers):
                input_var = program.global_block().create_var(
                    name="%s.trainer_%d" % (orig_var_name, trainer_id),
                    persistable=False,
                    type=v.type,
                    dtype=v.dtype,
                    shape=v.shape)
    return program


def add_optimizer_pass(program, config):
    def _is_opt_op_on_pserver(endpoint, op):
        param_names = [
            p.name for p in self.param_grad_ep_mapping[endpoint]["params"]
        ]
        if op.input("Param")[0] in param_names:
            return True
        else:
            for n in param_names:
                param = op.input("Param")[0]
                if _same_or_split_var(n, param) and n != param:
                    return True
            return False

    origin_program = config.get_origin_main_program
    ps_endpoint = config.get_ps_endpoint()

    opt_op_on_pserver = []
    # Iterate through the ops, and if an op and the optimize ops
    # which located on current pserver are in one set, then
    # append it into the sub program.
    global_ops = []
    # sparse grad name to param name
    sparse_grad_to_param = []

    optimize_ops = _get_optimize_ops(origin_program)

    for _, op in enumerate(optimize_ops):
        if _is_optimizer_op(op) and _is_opt_op_on_pserver(ps_endpoint, op):
            opt_op_on_pserver.append(op)

        def __append_optimize_op__(op, block, grad_to_block_id, merged_var,
                                   lr_ops):
            if _is_optimizer_op(op):
                _append_pserver_ops(block, op, ps_endpoint, grad_to_block_id,
                                    origin_program, merged_var,
                                    sparse_grad_to_param)
            elif op not in lr_ops:
                _append_pserver_non_opt_ops(block, op)

        def __clone_lr_op_sub_block__(op, program, lr_block):
            if not op.has_attr('sub_block'):
                return

            origin_block_desc = op.attr('sub_block')
            origin_block = origin_program.block(origin_block_desc.id)
            assert isinstance(origin_block, Block)
            # we put the new sub block to new block to follow the block
            # hierarchy of the original blocks
            new_sub_block = program._create_block(lr_block.idx)

            # clone vars
            for var in origin_block.vars:
                new_sub_block._clone_variable(var)

            # clone ops
            for origin_op in origin_block.ops:
                cloned_op = _clone_lr_op(program, new_sub_block, origin_op)
                # clone sub_block of op
                __clone_lr_op_sub_block__(cloned_op, program, new_sub_block)

            # reset the block of op
            op._set_attr('sub_block', new_sub_block)

        def _get_lr_ops():
            lr_ops = []
            block = origin_program.global_block()
            for index, op in enumerate(block.ops):
                role_id = int(op.attr(RPC_OP_ROLE_ATTR_NAME))
                if role_id == int(LR_SCHED_OP_ROLE_ATTR_VALUE) or \
                        role_id == int(LR_SCHED_OP_ROLE_ATTR_VALUE) | \
                        int(OPT_OP_ROLE_ATTR_VALUE):
                    lr_ops.append(op)
            return lr_ops

        # append lr decay ops to the child block if exists
        lr_ops = _get_lr_ops()
        lr_decay_block_id = -1
        optimize_blocks = []

        if len(lr_ops) > 0:
            lr_decay_block = program._create_block(program.num_blocks - 1)
            optimize_blocks.append(lr_decay_block)
            for _, op in enumerate(lr_ops):
                cloned_op = _append_pserver_non_opt_ops(lr_decay_block, op)
                # append sub blocks to pserver_program in lr_decay_op
                __clone_lr_op_sub_block__(cloned_op, program, lr_decay_block)
            lr_decay_block_id = lr_decay_block.idx

        # append op to the current block
        grad_to_block_id = []
        pre_block_idx = program.num_blocks - 1
        for idx, opt_op in enumerate(opt_op_on_pserver):
            per_opt_block = program._create_block(pre_block_idx)
            optimize_blocks.append(per_opt_block)
            optimize_target_param_name = opt_op.attr(OP_ROLE_VAR_ATTR_NAME)[0]
            # append grad merging ops before clip and weight decay
            # e.g. merge grad -> L2Decay op -> clip op -> optimize
            merged_var = None
            for _, op in enumerate(optimize_ops):
                # find the origin grad var before clipping/L2Decay,
                # merged_var should be the input var name of L2Decay
                grad_varname_for_block = op.attr(OP_ROLE_VAR_ATTR_NAME)[1]
                if op.attr(OP_ROLE_VAR_ATTR_NAME)[
                        0] == optimize_target_param_name:
                    merged_var = _append_pserver_grad_merge_ops(
                        per_opt_block, grad_varname_for_block, ps_endpoint,
                        grad_to_block_id, origin_program)
                    if merged_var:
                        break  # append optimize op once then append other ops.
            if merged_var:
                for _, op in enumerate(optimize_ops):
                    # optimizer is connected to itself
                    if op.attr(OP_ROLE_VAR_ATTR_NAME)[0] == optimize_target_param_name and \
                            op not in global_ops:
                        __append_optimize_op__(op, per_opt_block,
                                               grad_to_block_id, merged_var,
                                               lr_ops)

        # dedup grad to ids list
        grad_to_block_id = list(set(grad_to_block_id))
        # append global ops
        if global_ops:
            opt_state_block = program._create_block(program.num_blocks - 1)
            optimize_blocks.append(opt_state_block)
            for glb_op in global_ops:
                __append_optimize_op__(glb_op, opt_state_block,
                                       grad_to_block_id, None, lr_ops)

        if len(optimize_blocks) == 0:
            pre_block_idx = program.num_blocks - 1
            empty_block = program._create_block(pre_block_idx)
            optimize_blocks.append(empty_block)

    op = get_op_by_type(program.global_block(), "listen_and_serv")
    op._set_attr("optimize_blocks", optimize_blocks)
    op._set_attr("grad_to_block_id", grad_to_block_id)
    op._set_attr("sparse_grad_to_param", sparse_grad_to_param)
    op._set_attr("lr_decay_block_id", lr_decay_block_id)


def build_pserver_startup_program_pass(program, p_main_program, config):
    ps_endpoint = config.get_ps_endpoint()
    o_startup_program = config.get_origin_startup_program()
    program.random_seed = o_startup_program.random_seed

    params = self.param_grad_ep_mapping[ps_endpoint]["params"]

    def _get_splited_name_and_shape(varname):
        for idx, splited_param in enumerate(params):
            pname = splited_param.name
            if _same_or_split_var(pname, varname) and varname != pname:
                return pname, splited_param.shape
        return "", []

    # 1. create vars in pserver program to startup program
    pserver_vars = p_main_program.global_block().vars
    created_var_map = collections.OrderedDict()
    for _, var in six.iteritems(pserver_vars):
        tmpvar = program.global_block()._clone_variable(var)
        created_var_map[var.name] = tmpvar

    # 2. rename op outputs
    for op in o_startup_program.global_block().ops:
        new_outputs = collections.OrderedDict()
        # do not append startup op if var is not on this pserver
        op_on_pserver = False
        # TODO(gongwb): remove this line.
        if op.type not in ["recv", "fetch_barrier", "concat"]:
            for key in op.output_names:
                newname, _ = _get_splited_name_and_shape(op.output(key)[0])
                if newname:
                    op_on_pserver = True
                    new_outputs[key] = created_var_map[newname]
                elif op.output(key)[0] in pserver_vars:
                    op_on_pserver = True
                    new_outputs[key] = pserver_vars[op.output(key)[0]]

        if op_on_pserver:
            # most startup program ops have no inputs
            new_inputs = _get_input_map_from_op(pserver_vars, op)

            if op.type in [
                    "gaussian_random", "fill_constant", "uniform_random",
                    "truncated_gaussian_random"
            ]:
                op._set_attr("shape", list(new_outputs["Out"].shape))

            program.global_block().append_op(
                type=op.type,
                inputs=new_inputs,
                outputs=new_outputs,
                attrs=op.all_attrs())

    return program
