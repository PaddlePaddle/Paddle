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

from paddle.fluid.incubate.fleet.parameter_server.ir.public import get_param_grads
from paddle.fluid.incubate.fleet.parameter_server.ir.public import clone_variable
from paddle.fluid.incubate.fleet.parameter_server.ir.public import _get_optimize_ops
from paddle.fluid.incubate.fleet.parameter_server.ir.public import _is_optimizer_op
from paddle.fluid.incubate.fleet.parameter_server.ir.public import _is_optimizer_op


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
            if same_or_split_var(n, param) and n != param:
                return True
        return False


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
            self._append_pserver_ops(block, op, endpoint, grad_to_block_id,
                                     self.origin_program, merged_var,
                                     sparse_grad_to_param)
        elif op not in lr_ops:
            self._append_pserver_non_opt_ops(block, op)

    def __clone_lr_op_sub_block__(op, program, lr_block):
        if not op.has_attr('sub_block'):
            return

        origin_block_desc = op.attr('sub_block')
        origin_block = self.origin_program.block(origin_block_desc.id)
        assert isinstance(origin_block, Block)
        # we put the new sub block to new block to follow the block
        # hierarchy of the original blocks
        new_sub_block = program._create_block(lr_block.idx)

        # clone vars
        for var in origin_block.vars:
            new_sub_block._clone_variable(var)

        # clone ops
        for origin_op in origin_block.ops:
            cloned_op = self._clone_lr_op(program, new_sub_block, origin_op)
            # clone sub_block of op
            __clone_lr_op_sub_block__(cloned_op, program, new_sub_block)

        # reset the block of op
        op._set_attr('sub_block', new_sub_block)

    # append lr decay ops to the child block if exists
    lr_ops = self._get_lr_ops()
    # record optimize blocks and we can run them on pserver parallel
    optimize_blocks = []

    lr_decay_block_id = -1
    if len(lr_ops) > 0:
        lr_decay_block = pserver_program._create_block(
            pserver_program.num_blocks - 1)
        optimize_blocks.append(lr_decay_block)
        for _, op in enumerate(lr_ops):
            cloned_op = self._append_pserver_non_opt_ops(lr_decay_block, op)
            # append sub blocks to pserver_program in lr_decay_op
            __clone_lr_op_sub_block__(cloned_op, pserver_program,
                                      lr_decay_block)
        lr_decay_block_id = lr_decay_block.idx

    # append op to the current block
    grad_to_block_id = []
    pre_block_idx = pserver_program.num_blocks - 1
    for idx, opt_op in enumerate(opt_op_on_pserver):
        per_opt_block = pserver_program._create_block(pre_block_idx)
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
                    log("append opt op: ", op.type, op.input_arg_names,
                        merged_var)
                    __append_optimize_op__(op, per_opt_block, grad_to_block_id,
                                           merged_var, lr_ops)

    # dedup grad to ids list
    grad_to_block_id = list(set(grad_to_block_id))
    # append global ops
    if global_ops:
        opt_state_block = pserver_program._create_block(
            pserver_program.num_blocks - 1)
        optimize_blocks.append(opt_state_block)
        for glb_op in global_ops:
            __append_optimize_op__(glb_op, opt_state_block, grad_to_block_id,
                                   None, lr_ops)

    if len(optimize_blocks) == 0:
        logging.warn("pserver [" + str(endpoint) + "] has no optimize block!!")
        pre_block_idx = pserver_program.num_blocks - 1
        empty_block = pserver_program._create_block(pre_block_idx)
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
