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
from paddle.fluid import initializer
from paddle.fluid.framework import Block

from paddle.fluid.incubate.fleet.parameter_server.ir.public import get_param_grads
from paddle.fluid.incubate.fleet.parameter_server.ir.public import clone_variable
from paddle.fluid.incubate.fleet.parameter_server.ir.public import _get_optimize_ops
from paddle.fluid.incubate.fleet.parameter_server.ir.public import _is_optimizer_op
from paddle.fluid.incubate.fleet.parameter_server.ir.public import DistributedMode
from paddle.fluid.incubate.fleet.parameter_server.ir.public import ServerRuntimeConfig

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


def _same_or_split_var(p_name, var_name):
    return p_name == var_name or p_name.startswith(var_name + ".block")


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
        if key in [
                "Param", "Grad", "LearningRate", "Beta1Tensor", "Beta2Tensor"
        ]:
            continue
        var = origin_program.global_block().vars[opt_op.input(key)[0]]
        param_var = new_inputs["Param"]
        # update accumulator variable shape
        new_shape = self._get_optimizer_input_shape(opt_op.type, key, var.shape,
                                                    param_var.shape)
        tmpvar = pserver_block.create_var(
            name=var.name,
            persistable=var.persistable,
            dtype=var.dtype,
            shape=new_shape)
        new_inputs[key] = tmpvar

    # change output's ParamOut variable
    outputs = self._get_output_map_from_op(
        self.origin_program.global_block().vars, opt_op)
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
                "Not supported optimizer for distributed training: %s" %
                op_type)
        return orig_shape


def add_dist_info_pass(program, origin_program):
    program.random_seed = origin_program.random_seed
    program._copy_dist_param_info_from(origin_program)
    return program


def add_receive_vars_pass(program, mode, trainers, role_id, pserver_endpoints):
    params, grads = get_param_grad_by_endpoint(role_id, pserver_endpoints)

    for param in params:
        clone_variable(program.global_block(), param)

    for grad in grads:
        # create vars for each trainer in global scope, so
        # we don't need to create them when grad arrives.
        # change client side var name to origin name by
        # removing ".trainer_%d" suffix
        suff_idx = grad.name.find(".trainer_")
        if suff_idx >= 0:
            orig_var_name = grad.name[:suff_idx]
        else:
            orig_var_name = grad.name

        if mode == DistributedMode.SYNC:
            for trainer_id in range(trainers):
                program.global_block().create_var(
                    name="%s.trainer_%d" % (orig_var_name, trainer_id),
                    persistable=False,
                    type=grad.type,
                    dtype=grad.dtype,
                    shape=grad.shape)
        else:
            program.global_block().create_var(
                name=orig_var_name,
                persistable=True,
                type=grad.type,
                dtype=grad.dtype,
                shape=grad.shape)


def add_listen_and_serve_pass(program, origin_program, pserver_endpoints):
    optimize_ops = _get_optimize_ops(origin_program)
    opt_op_on_pserver = []
    for _, op in enumerate(optimize_ops):
        if _is_optimizer_op(op) and _is_opt_op_on_pserver(pserver_endpoints,
                                                          op):
            opt_op_on_pserver.append(op)

    global_ops = []

    # sparse grad name to param name
    sparse_grad_to_param = []

    def __append_optimize_op__(op, block, grad_to_block_id, merged_var, lr_ops):
        if _is_optimizer_op(op):
            _append_pserver_ops(block, op, pserver_endpoints, grad_to_block_id,
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

    def _get_lr_ops(self):
        lr_ops = []
        block = self.origin_program.global_block()
        for index, op in enumerate(block.ops):
            role_id = int(op.attr(RPC_OP_ROLE_ATTR_NAME))
            if role_id == int(LR_SCHED_OP_ROLE_ATTR_VALUE) or \
                    role_id == int(LR_SCHED_OP_ROLE_ATTR_VALUE) | \
                    int(OPT_OP_ROLE_ATTR_VALUE):
                if self.sync_mode == False and op.type == 'increment':
                    inputs = _get_input_map_from_op(
                        origin_program.global_block().vars, op)
                    outputs = self._get_output_map_from_op(
                        self.origin_program.global_block().vars, op)
                    for key in outputs:
                        counter_var = outputs[key]
                    all_trainer_counter_inputs = [
                        self.origin_program.global_block().create_var(
                            name="%s.trainer_%d" % (counter_var.name, id_),
                            type=counter_var.type,
                            shape=counter_var.shape,
                            dtype=counter_var.dtype,
                            persistable=counter_var.persistable)
                        for id_ in range(self.trainer_num)
                    ]
                    for i, op in enumerate(self.startup_program.global_block()
                                           .ops):
                        if op.type == 'fill_constant':
                            for key in op.output_names:
                                if len(op.output(key)) == 1 and op.output(key)[
                                        0] == counter_var.name:
                                    self.startup_program.global_block().ops[
                                        i]._set_attr(
                                            'value',
                                            float(0.0 - self.trainer_num))
                    for var in all_trainer_counter_inputs:
                        if var.name == "%s.trainer_%d" % (counter_var.name,
                                                          self.trainer_id):
                            self.counter_var = var
                        self.startup_program.global_block().create_var(
                            name=var.name,
                            type=var.type,
                            dtype=var.dtype,
                            shape=var.shape,
                            persistable=var.persistable,
                            initializer=initializer.Constant(1))
                    op_role_attr_name = core.op_proto_and_checker_maker.kOpRoleAttrName(
                    )
                    block._remove_op(index)
                    op = block._insert_op(
                        index,
                        type='sum',
                        inputs={'X': all_trainer_counter_inputs},
                        outputs=outputs,
                        attrs={op_role_attr_name: LR_SCHED_OP_ROLE_ATTR_VALUE})
                lr_ops.append(op)
        return lr_ops

    def _clone_lr_op(program, block, op):
        inputs = _get_input_map_from_op(origin_program.global_block().vars, op)
        for key, varlist in six.iteritems(inputs):
            if not isinstance(varlist, list):
                varlist = [varlist]
            for var in varlist:
                if var not in program.global_block().vars:
                    block._clone_variable(var)

        outputs = _get_output_map_from_op(origin_program.global_block().vars,
                                          op)
        for key, varlist in six.iteritems(outputs):
            if not isinstance(varlist, list):
                varlist = [varlist]
            for var in varlist:
                if var not in program.global_block().vars:
                    block._clone_variable(var)

        return block.append_op(
            type=op.type, inputs=inputs, outputs=outputs, attrs=op.all_attrs())

    def _get_varname_parts(varname):
        # returns origin, blockid, trainerid
        orig_var_name = ""
        trainer_part = ""
        block_part = ""
        trainer_idx = varname.find(".trainer_")
        if trainer_idx >= 0:
            trainer_part = varname[trainer_idx + 1:]
        else:
            trainer_idx = len(varname)
        block_index = varname.find(".block")
        if block_index >= 0:
            block_part = varname[block_index + 1:trainer_idx]
        else:
            block_index = len(varname)
        orig_var_name = varname[0:min(block_index, trainer_idx)]
        return orig_var_name, block_part, trainer_part

    def _orig_varname(varname):
        orig, _, _ = _get_varname_parts(varname)
        return orig

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

    def _append_pserver_grad_merge_ops(optimize_block, grad_varname_for_block,
                                       endpoint, grad_to_block_id,
                                       origin_program):
        program = optimize_block.program
        pserver_block = program.global_block()
        grad_block = None
        for g in self.param_grad_ep_mapping[endpoint]["grads"]:
            if _orig_varname(g.name) == _orig_varname(grad_varname_for_block):
                grad_block = g
                break
        if not grad_block:
            # do not append this op if current endpoint
            # is not dealing with this grad block
            return None
        orig_varname, block_name, trainer_name = _get_varname_parts(
            grad_block.name)
        if block_name:
            merged_var_name = '.'.join([orig_varname, block_name])
        else:
            merged_var_name = orig_varname

        merged_var = pserver_block.vars[merged_var_name]
        grad_to_block_id.append(merged_var.name + ":" + str(optimize_block.idx))
        if mode == DistributedMode.SYNC and trainers > 1:
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

    def _append_pserver_non_opt_ops(optimize_block, opt_op):
        program = optimize_block.program
        # Append the ops for parameters that do not need to be optimized/updated
        inputs = _get_input_map_from_op(origin_program.global_block().vars,
                                        opt_op)
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

    # append lr decay ops to the child block if exists
    lr_ops = _get_lr_ops()
    # record optimize blocks and we can run them on pserver parallel
    optimize_blocks = []

    lr_decay_block_id = -1
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
        for _, op in enumerate(self.optimize_ops):
            # find the origin grad var before clipping/L2Decay,
            # merged_var should be the input var name of L2Decay
            grad_varname_for_block = op.attr(OP_ROLE_VAR_ATTR_NAME)[1]
            if op.attr(OP_ROLE_VAR_ATTR_NAME)[0] == optimize_target_param_name:
                merged_var = self._append_pserver_grad_merge_ops(
                    per_opt_block, grad_varname_for_block, endpoint,
                    grad_to_block_id, self.origin_program)
                if merged_var:
                    break  # append optimize op once then append other ops.
        if merged_var:
            for _, op in enumerate(self.optimize_ops):
                # optimizer is connected to itself
                if op.attr(OP_ROLE_VAR_ATTR_NAME)[0] == optimize_target_param_name and \
                        op not in global_ops:
                    __append_optimize_op__(op, per_opt_block, grad_to_block_id,
                                           merged_var, lr_ops)

    # dedup grad to ids list
    grad_to_block_id = list(set(grad_to_block_id))
    # append global ops
    if global_ops:
        opt_state_block = program._create_block(program.num_blocks - 1)
        optimize_blocks.append(opt_state_block)
        for glb_op in global_ops:
            __append_optimize_op__(glb_op, opt_state_block, grad_to_block_id,
                                   None, lr_ops)

    if len(optimize_blocks) == 0:
        pre_block_idx = program.num_blocks - 1
        empty_block = program._create_block(pre_block_idx)
        optimize_blocks.append(empty_block)

    # In some case, some parameter server will have no parameter to optimize
    # So we give an empty optimize block to parameter server.
    attrs = {
        "optimize_blocks": optimize_blocks,
        "endpoint": endpoint,
        "pserver_id": self.pserver_endpoints.index(endpoint),
        "Fanin": self.trainer_num,
        "distributed_mode": self.distributed_mode,
        "grad_to_block_id": grad_to_block_id,
        "sparse_grad_to_param": sparse_grad_to_param,
        "lr_decay_block_id": lr_decay_block_id,
        "rpc_get_thread_num": self.server_config._rpc_get_thread_num,
        "rpc_send_thread_num": self.server_config._rpc_send_thread_num,
        "rpc_prefetch_thread_num": self.server_config._rpc_prefetch_thread_num
    }

    # step5 append the listen_and_serv op
    program.global_block().append_op(
        type="listen_and_serv", inputs={'X': []}, outputs={}, attrs=attrs)

    return program
