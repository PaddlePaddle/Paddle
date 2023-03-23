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
"""
Transpile the program to distributed data-parallelism programs.
The main_program will be transformed to use a remote parameter server
to do parameter optimization. And the optimization graph will be put
into a parameter server program.

Use different methods to split trainable variables to different
parameter servers.

Steps to transpile trainer:
1. split variable to multiple blocks, aligned by product(dim[1:]) (width).
2. rename splited grad variables to add trainer_id suffix ".trainer_%d".
3. modify trainer program add split_op to each grad variable.
4. append send_op to send splited variables to server and fetch
    params(splited blocks or origin param) from server.
5. append concat_op to merge splited blocks to update local weights.

Steps to transpile pserver:
1. create new program for parameter server.
2. create params and grad variables that assigned to current server instance.
3. create a sub-block in the server side program
4. append ops that should run on current server instance.
5. add listen_and_serv op
"""

from __future__ import print_function

import math

from ps_dispatcher import RoundRobin, HashName, PSDispatcher
from .. import core, framework
from ..framework import Program, default_main_program, \
                        default_startup_program, \
                        Variable, Parameter, grad_var_name
from details import *

LOOKUP_TABLE_TYPE = "lookup_table"
LOOKUP_TABLE_GRAD_TYPE = "lookup_table_grad"
OP_ROLE_VAR_ATTR_NAME = core.op_proto_and_checker_maker.kOpRoleVarAttrName()
RPC_OP_ROLE_ATTR_NAME = op_role_attr_name = core.op_proto_and_checker_maker.kOpRoleAttrName(
)
RPC_OP_ROLE_ATTR_VALUE = core.op_proto_and_checker_maker.OpRole.RPC


class VarBlock:
    def __init__(self, varname, offset, size):
        self.varname = varname
        # NOTE: real offset is offset * size
        self.offset = offset
        self.size = size

    def __str__(self):
        return "%s:%d:%d" % (self.varname, self.offset, self.size)


def same_or_split_var(p_name, var_name):
    return p_name == var_name or p_name.startswith(var_name + ".block")


def split_variable(var_list, service_count, min_block_size=8192):
    """
    We may need to split dense tensor to one or more blocks and put
    them equally onto parameter server. One block is a sub-tensor
    aligned by dim[0] of the tensor.

    We need to have a minimal block size so that the calculations in
    the parameter server side can gain better performance. By default
    minimum block size 8K elements (maybe 16bit or 32bit or 64bit). 

    Args:
        var_list (list): List of variables.
        service_count (int): Numel of pserver services. A pserver may have two
            or more listening ports.
        min_block_size (int): Minimum splitted block size.
    Returns:
        blocks (list[(varname, block_id, current_block_size)]): A list 
            of VarBlocks. Each VarBlock specifies a shard of the var.
    """
    blocks = []
    for var in var_list:
        split_count = service_count
        var_numel = reduce(lambda x, y: x * y, var.shape)
        max_pserver_count = int(math.floor(var_numel / float(min_block_size)))
        if max_pserver_count == 0:
            max_pserver_count = 1
        if max_pserver_count < service_count:
            split_count = max_pserver_count
        block_size = int(math.ceil(var_numel / float(split_count)))

        if len(var.shape) >= 2:
            # align by dim1(width)
            dim1 = reduce(lambda x, y: x * y, var.shape[1:])
            remains = block_size % dim1
            if remains != 0:
                block_size += dim1 - remains
        # update split_count after aligning
        split_count = int(math.ceil(var_numel / float(block_size)))
        for block_id in xrange(split_count):
            curr_block_size = min(block_size, var_numel - (
                (block_id) * block_size))
            block = VarBlock(var.name, block_id, curr_block_size)
            blocks.append(str(block))
    return blocks


class DistributeTranspiler:
    def _has_distributed_lookup_table(self):
        # process lookup_table_op
        # 1. check all lookup_table_op is distributed
        # 2. check all lookup_table_op share the same table.
        distributed_lookup_table_ops = []
        # support only one distributed_lookup_table now
        self.table_name = None
        for op in self.origin_program.global_block().ops:
            if op.type == LOOKUP_TABLE_TYPE:
                if op.attrs['is_distributed'] is True:
                    if self.table_name is None:
                        self.table_name = op.input("W")[0]
                    if self.table_name != op.input("W")[0]:
                        raise RuntimeError("all distributed lookup_table_ops"
                                           " should have only one table")
                    distributed_lookup_table_ops.append(op)
                else:
                    if self.table_name is not None:
                        assert op.input("W")[0] != self.table_name

        return len(distributed_lookup_table_ops) > 0

    def _update_dist_lookup_table_vars(self, param_list, grad_list,
                                       params_grads):
        # TODO(wuyi): put find a way to put dist lookup table stuff all together.
        # update self.table_param_grad and self.trainer_side_table_grad_list
        program = self.origin_program
        if self.has_distributed_lookup_table:
            param_list = [
                param for param in param_list if param.name != self.table_name
            ]
            grad_list = [
                grad for grad in grad_list
                if grad.name != grad_var_name(self.table_name)
            ]
            self.table_param_grad = [
                param_grad for param_grad in params_grads
                if param_grad[0].name == self.table_name
            ][0]
            table_grad_var = self.table_param_grad[1]
            if self.sync_mode:
                self.trainer_side_table_grad_list = [
                    program.global_block().create_var(
                        name="%s.trainer_%d.pserver_%d" %
                        (table_grad_var.name, self.trainer_id, index),
                        type=table_grad_var.type,
                        shape=table_grad_var.shape,
                        dtype=table_grad_var.dtype)
                    for index in range(len(self.pserver_endpoints))
                ]
            else:
                self.trainer_side_table_grad_list = [
                    program.global_block().create_var(
                        name="%s.pserver_%d" % (table_grad_var.name, index),
                        type=table_grad_var.type,
                        shape=table_grad_var.shape,
                        dtype=table_grad_var.dtype)
                    for index in range(len(self.pserver_endpoints))
                ]

    def _init_splited_vars(self, split_method):
        # update these mappings for further transpile:
        # 1. param_var_mapping: param var name -> [splited params vars]
        # 2. grad_var_mapping: grad var name -> [splited grads vars]
        # 3. grad_param_mapping: grad.blockx -> param.blockx
        # 4. param_grad_ep_mapping: ep -> {"params": [], "grads": []}

        param_list = []
        grad_list = []
        param_grad_set = set()
        for p, g in self.params_grads:
            # skip parameter marked not trainable
            if type(p) == Parameter and p.trainable == False:
                continue
            if p.name not in param_grad_set:
                param_list.append(p)
                param_grad_set.add(p.name)
            if g.name not in param_grad_set:
                grad_list.append(g)
                param_grad_set.add(g.name)

        self._update_dist_lookup_table_vars(param_list, grad_list,
                                            self.params_grads)

        grad_blocks = split_variable(grad_list, len(self.pserver_endpoints))
        param_blocks = split_variable(param_list, len(self.pserver_endpoints))
        assert (len(grad_blocks) == len(param_blocks))
        # origin_varname -> [splited_var]
        self.param_var_mapping = self._create_vars_from_blocklist(
            self.origin_program, param_blocks)
        self.grad_var_mapping = self._create_vars_from_blocklist(
            self.origin_program,
            grad_blocks,
            add_trainer_suffix=self.trainer_num > 1)
        self.grad_param_mapping = dict()
        for g, p in zip(grad_blocks, param_blocks):
            g_name, g_bid, _ = g.split(":")
            p_name, p_bid, _ = p.split(":")
            self.grad_param_mapping[self.grad_var_mapping[g_name][int(g_bid)]] =  \
                    self.param_var_mapping[p_name][int(p_bid)]

        # create mapping of endpoint -> split var to create pserver side program
        self.param_grad_ep_mapping = dict()
        [
            self.param_grad_ep_mapping.update({
                ep: {
                    "params": [],
                    "grads": []
                }
            }) for ep in self.pserver_endpoints
        ]

    def transpile(self,
                  trainer_id,
                  program=None,
                  pservers="127.0.0.1:6174",
                  trainers=1,
                  split_method=RoundRobin,
                  sync_mode=True):
        """
        :param trainer_id: one unique id for each trainer in a job.
        :type trainer_id: int
        :param program: program to transpile, default is default_main_program
        :type program: Program
        :param pservers: parameter server endpoints like "m1:6174,m2:6174"
        :type pservers: string
        :param trainers: total number of workers/trainers in the job
        :type trainers: int
        :param split_method: A function to determin how to split variables
            to different servers equally.
        :type split_method: function
        :type sync_mode: boolean default True
        :param sync_mode: if sync_mode is set True, it means that dist transpiler
        will transpile the program into sync_mode pserver and trainer program.
        """
        assert (split_method.__bases__[0] == PSDispatcher)
        if program is None:
            program = default_main_program()
        self.origin_program = program
        self.trainer_num = trainers
        self.sync_mode = sync_mode
        self.trainer_id = trainer_id
        pserver_endpoints = pservers.split(",")
        self.pserver_endpoints = pserver_endpoints
        self.optimize_ops, self.params_grads = self._get_optimize_pass()

        ps_dispatcher = split_method(self.pserver_endpoints)
        self.has_distributed_lookup_table = self._has_distributed_lookup_table()

        # split and create vars, then put splited vars in dicts for later use.
        self._init_splited_vars(split_method)

        # step 3.1: insert send op to send gradient vars to parameter servers
        ps_dispatcher.reset()
        send_vars = []
        for orig_varname, splited_vars in self.grad_var_mapping.items():
            eplist = ps_dispatcher.dispatch(splited_vars)
            if len(splited_vars) == 1:
                orig_varname = splited_vars[0].name
                index = find_op_by_output_arg(program.global_block(),
                                              orig_varname)
            elif len(splited_vars) > 1:
                orig_var = program.global_block().vars[orig_varname]
                index = find_op_by_output_arg(program.global_block(),
                                              orig_varname)
                self._insert_split_op(program, orig_var, index, splited_vars)
                index += 1
            else:
                AssertionError("Can not insert the send op by original "
                               "variable name :", orig_varname)

            program.global_block().insert_op(
                index=index + 1,
                type="send_vars",
                inputs={"X": splited_vars},
                outputs={},
                attrs={
                    "epmap": eplist,
                    RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE
                })
            for _, var in enumerate(splited_vars):
                send_vars.append(var)

        if self.sync_mode:
            program.global_block().append_op(
                type="send_barrier",
                inputs={},
                outputs={},
                attrs={
                    "endpoints": pserver_endpoints,
                    "sync_mode": self.sync_mode,
                    RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE
                })

        # step 3.2: insert recv op to receive parameters from parameter server
        recv_vars = []
        for _, var in enumerate(send_vars):
            recv_vars.append(self.grad_param_mapping[var])
        ps_dispatcher.reset()
        eplist = ps_dispatcher.dispatch(recv_vars)

        for i, ep in enumerate(eplist):
            self.param_grad_ep_mapping[ep]["params"].append(recv_vars[i])
            self.param_grad_ep_mapping[ep]["grads"].append(send_vars[i])
        # step4: Concat the parameters splits together after recv.
        for varname, splited_var in self.param_var_mapping.iteritems():
            eps = []
            for var in splited_var:
                index = [v.name for v in recv_vars].index(var.name)
                eps.append(eplist[index])

            program.global_block().append_op(
                type="recv",
                inputs={},
                outputs={"Out": splited_var},
                attrs={
                    "epmap": eps,
                    RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE
                })

        program.global_block().append_op(
            type="fetch_barrier",
            inputs={},
            outputs={},
            attrs={
                "endpoints": pserver_endpoints,
                RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE
            })

        for varname, splited_var in self.param_var_mapping.iteritems():
            if len(splited_var) <= 1:
                continue
            orig_param = program.global_block().vars[varname]
            program.global_block().append_op(
                type="concat",
                inputs={"X": splited_var},
                outputs={"Out": [orig_param]},
                attrs={"axis": 0})

        if self.has_distributed_lookup_table:
            self._replace_lookup_table_op_with_prefetch(program,
                                                        pserver_endpoints)
            self._split_table_grad_and_add_send_vars(program, pserver_endpoints)

    def get_trainer_program(self):
        # remove optimize ops and add a send op to main_program
        delete_ops(self.origin_program.global_block(), self.optimize_ops)
        # FIXME(typhoonzero): serialize once will fix error occurs when clone.
        self.origin_program.__str__()
        return self.origin_program

    def get_pserver_program(self, endpoint):
        """
        Get pserver side program using the endpoint.
        TODO(panyx0718): Revisit this assumption. what if #blocks > #pservers.
        NOTE: assume blocks of the same variable is not distributed
        on the same pserver, only change param/grad varnames for
        trainers to fetch.
        """
        # step1
        pserver_program = Program()
        # step2: Create vars to receive vars at parameter servers.
        recv_inputs = []
        for v in self.param_grad_ep_mapping[endpoint]["params"]:
            self._clone_var(pserver_program.global_block(), v)
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
            single_trainer_var = \
                pserver_program.global_block().create_var(
                    name=orig_var_name,
                    persistable=True,
                    type=v.type,
                    dtype=v.dtype,
                    shape=v.shape)
            if self.sync_mode and self.trainer_num > 1:
                for trainer_id in xrange(self.trainer_num):
                    var = pserver_program.global_block().create_var(
                        name="%s.trainer_%d" % (orig_var_name, trainer_id),
                        persistable=False,
                        type=v.type,
                        dtype=v.dtype,
                        shape=v.shape)
                    recv_inputs.append(var)
            else:
                recv_inputs.append(single_trainer_var)

        # step 3
        # Create a union-find data structure from optimize ops,
        # If two ops are connected, we could add these two ops
        # into one set.
        ufind = self._create_ufind(self.optimize_ops)
        # step 3.2
        # Iterate through the ops and append optimize op which
        # located on current pserver
        opt_op_on_pserver = []
        for _, op in enumerate(self.optimize_ops):
            if self._is_optimizer_op(op) and self._is_opt_op_on_pserver(
                    endpoint, op):
                opt_op_on_pserver.append(op)
        # step 3.3
        # Iterate through the ops, and if an op and the optimize ops
        # which located on current pserver are in one set, then
        # append it into the sub program.

        global_ops = []
        # HACK: optimization global ops only used to scale beta1 and beta2
        # replace it with dependency engine.
        for op in self.optimize_ops:
            if self._is_adam_connected_op(op):
                global_ops.append(op)

        def __append_optimize_op__(op, block, grad_to_block_id, merged_var):
            if self._is_optimizer_op(op):
                self._append_pserver_ops(block, op, endpoint, grad_to_block_id,
                                         self.origin_program, merged_var)
            else:
                self._append_pserver_non_opt_ops(block, op, endpoint)

        def __op_have_grad_input__(op):
            for varname in op.input_arg_names:
                if varname.find("@GRAD") >= 0:
                    return varname
            return ""

        # append lr decay ops to the child block if exists
        lr_ops = self._get_lr_ops()
        if len(lr_ops) > 0:
            lr_decay_block = pserver_program.create_block(
                pserver_program.num_blocks - 1)
            for _, op in enumerate(lr_ops):
                self._append_pserver_non_opt_ops(lr_decay_block, op, endpoint)

        # append op to the current block
        grad_to_block_id = []
        pre_block_idx = pserver_program.num_blocks - 1
        for idx, opt_op in enumerate(opt_op_on_pserver):
            per_opt_block = pserver_program.create_block(pre_block_idx)
            # append grad merging ops before clip and weight decay
            for _, op in enumerate(self.optimize_ops):
                # find the origin @GRAD var before clipping
                grad_varname_for_block = __op_have_grad_input__(op)
                if ufind.is_connected(op, opt_op) and grad_varname_for_block:
                    merged_var = self._append_pserver_grad_merge_ops(
                        per_opt_block, grad_varname_for_block, endpoint,
                        grad_to_block_id, self.origin_program)
            for _, op in enumerate(self.optimize_ops):
                # optimizer is connected to itself
                if ufind.is_connected(op, opt_op) and op not in global_ops:
                    __append_optimize_op__(op, per_opt_block, grad_to_block_id,
                                           merged_var)

        # append global ops
        if global_ops:
            opt_state_block = pserver_program.create_block(
                pserver_program.num_blocks - 1)
            for glb_op in global_ops:
                __append_optimize_op__(glb_op, opt_state_block,
                                       grad_to_block_id, None)

        # process distributed lookup_table
        prefetch_block = None
        if self.has_distributed_lookup_table:
            pserver_index = self.pserver_endpoints.index(endpoint)
            table_opt_block = self._create_table_optimize_block(
                pserver_index, pserver_program, pre_block_idx, grad_to_block_id)
            prefetch_block = self._create_prefetch_block(
                pserver_index, pserver_program, table_opt_block)

        # NOTE: if has_distributed_lookup_table is False, then prefetch_block will
        # not be executed, so it's safe to use optimize_block to hold the place
        if self.has_distributed_lookup_table:
            assert prefetch_block is not None
        else:
            assert prefetch_block is None
            prefetch_block = pserver_program.global_block()

        # step5 append the listen_and_serv op
        pserver_program.global_block().append_op(
            type="listen_and_serv",
            inputs={'X': recv_inputs},
            outputs={},
            attrs={
                "OptimizeBlock": pserver_program.block(1),
                "endpoint": endpoint,
                "Fanin": self.trainer_num,
                "PrefetchBlock": prefetch_block,
                "sync_mode": self.sync_mode,
                "grad_to_block_id": grad_to_block_id
            })

        pserver_program.sync_with_cpp()
        return pserver_program

    def get_startup_program(self, endpoint, pserver_program):
        """
        Get startup program for current parameter server.
        Modify operator input variables if there are variables that
        were split to several blocks.
        """
        s_prog = Program()
        orig_s_prog = default_startup_program()
        params = self.param_grad_ep_mapping[endpoint]["params"]

        def _get_splited_name_and_shape(varname):
            for idx, splited_param in enumerate(params):
                pname = splited_param.name
                if same_or_split_var(pname, varname) and varname != pname:
                    return pname, splited_param.shape
            return "", []

        # 1. create vars in pserver program to startup program
        pserver_vars = pserver_program.global_block().vars
        created_var_map = dict()
        for _, var in pserver_vars.iteritems():
            tmpvar = s_prog.global_block().clone_variable(var)
            created_var_map[var.name] = tmpvar

        # 2. rename op outputs
        for op in orig_s_prog.global_block().ops:
            new_inputs = dict()
            new_outputs = dict()
            # do not append startup op if var is not on this pserver
            op_on_pserver = False
            for key in op.output_names:
                newname, _ = _get_splited_name_and_shape(op.output(key)[0])
                if newname:
                    op_on_pserver = True
                    new_outputs[key] = created_var_map[newname]
                elif op.output(key)[0] in pserver_vars:
                    op_on_pserver = True
                    new_outputs[key] = pserver_vars[op.output(key)[0]]

            # most startup program ops have no inputs
            new_inputs = self._get_input_map_from_op(pserver_vars, op)

            if op_on_pserver:
                if op.type in [
                        "gaussian_random", "fill_constant", "uniform_random"
                ]:
                    op.attrs["shape"] = new_outputs["Out"].shape
                s_prog.global_block().append_op(
                    type=op.type,
                    inputs=new_inputs,
                    outputs=new_outputs,
                    attrs=op.attrs)
        return s_prog

    # ====================== private transpiler functions =====================

    # transpiler function for dis lookup_table
    def _replace_lookup_table_op_with_prefetch(self, program,
                                               pserver_endpoints):
        # 1. replace lookup_table_op with split_ids_op -> prefetch_op -> sum_op
        self.prefetch_input_vars = None
        self.prefetch_output_vars = None

        continue_search_lookup_table_op = True
        while continue_search_lookup_table_op:
            continue_search_lookup_table_op = False
            all_ops = program.global_block().ops
            for op in all_ops:
                if op.type == LOOKUP_TABLE_TYPE:
                    continue_search_lookup_table_op = True

                    op_index = list(all_ops).index(op)
                    ids_name = op.input("Ids")
                    out_name = op.output("Out")

                    if self.prefetch_input_vars is None:
                        ids_var = program.global_block().vars[ids_name[0]]
                        self.prefetch_input_vars = self.create_splited_vars(
                            source_var=ids_var,
                            block=program.global_block(),
                            tag="_prefetch_in_")
                    if self.prefetch_output_vars is None:
                        out_var = program.global_block().vars[out_name[0]]
                        self.prefetch_output_vars = self.create_splited_vars(
                            source_var=out_var,
                            block=program.global_block(),
                            tag="_prefetch_out_")

                    # insert split_ids_op
                    program.global_block().insert_op(
                        index=op_index,
                        type="split_ids",
                        inputs={
                            'Ids': [
                                program.global_block().vars[varname]
                                for varname in ids_name
                            ]
                        },
                        outputs={"Out": self.prefetch_input_vars})

                    # insert prefetch_op
                    program.global_block().insert_op(
                        index=op_index + 1,
                        type="prefetch",
                        inputs={'X': self.prefetch_input_vars},
                        outputs={"Out": self.prefetch_output_vars},
                        attrs={
                            "epmap": pserver_endpoints,
                            RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE
                        })

                    # insert concat_op
                    program.global_block().insert_op(
                        index=op_index + 2,
                        type="concat",
                        inputs={'X': self.prefetch_output_vars},
                        outputs={
                            "Out": [
                                program.global_block().vars[varname]
                                for varname in out_name
                            ]
                        },
                        attrs={"axis": 0})

                    # delete lookup_table_op
                    delete_ops(program.global_block(), [op])
                    # break for loop
                    break

    def _split_table_grad_and_add_send_vars(self, program, pserver_endpoints):
        # 2. add split_ids_op and send_vars_op to send gradient to pservers
        # there should only be one table_name
        all_ops = program.global_block().ops
        table_grad_name = grad_var_name(self.table_name)
        for op in all_ops:
            if table_grad_name in op.output_arg_names:
                op_index = list(all_ops).index(op)
                # insert split_ids_op
                program.global_block().insert_op(
                    index=op_index + 1,
                    type="split_ids",
                    inputs={
                        'Ids': [program.global_block().vars[table_grad_name]]
                    },
                    outputs={"Out": self.trainer_side_table_grad_list})
                program.global_block().insert_op(
                    index=op_index + 2,
                    type="send_vars",
                    inputs={'X': self.trainer_side_table_grad_list},
                    outputs={},
                    attrs={
                        "sync_send": True,
                        "epmap": pserver_endpoints,
                        RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE
                    })
                break

    def _create_prefetch_block(self, pserver_index, pserver_program,
                               optimize_block):
        # STEP: create prefetch block
        table_var = pserver_program.global_block().vars[self.table_name]
        prefetch_block = pserver_program.create_block(optimize_block.idx)
        trainer_ids = self.prefetch_input_vars[pserver_index]
        pserver_ids = pserver_program.global_block().create_var(
            name=trainer_ids.name,
            type=trainer_ids.type,
            shape=trainer_ids.shape,
            dtype=trainer_ids.dtype)
        trainer_out = self.prefetch_output_vars[pserver_index]
        pserver_out = pserver_program.global_block().create_var(
            name=trainer_out.name,
            type=trainer_out.type,
            shape=trainer_out.shape,
            dtype=trainer_out.dtype)
        prefetch_block.append_op(
            type="lookup_sparse_table",
            inputs={'Ids': pserver_ids,
                    "W": table_var},
            outputs={"Out": pserver_out},
            attrs={
                "is_sparse": True,  # has no effect on lookup_table op
                "is_distributed": True,
                "padding_idx": -1
            })
        return prefetch_block

    def _create_table_optimize_block(self, pserver_index, pserver_program,
                                     pre_block_idx, grad_to_block_id):
        # STEP: create table optimize block
        # create table param and grad var in pserver program
        origin_param_var = self.origin_program.global_block().vars[
            self.table_name]
        param_var = pserver_program.global_block().create_var(
            name=origin_param_var.name,
            shape=origin_param_var.shape,
            dtype=origin_param_var.dtype,
            type=core.VarDesc.VarType.SELECTED_ROWS,
            persistable=True)
        # parameter must be selected rows
        param_var.desc.set_type(core.VarDesc.VarType.SELECTED_ROWS)
        grad_var = pserver_program.global_block().clone_variable(
            self.origin_program.global_block().vars[grad_var_name(
                self.table_name)])

        # create table optimize block in pserver program
        table_opt_op = [
            op for op in self.optimize_ops
            if op.input("Param")[0] == self.table_name
        ][0]
        table_opt_block = pserver_program.create_block(pre_block_idx)
        # only support sgd now
        assert table_opt_op.type == "sgd"

        if self.sync_mode:
            # create grad vars in pserver program
            table_grad_var = self.table_param_grad[1]
            pserver_side_table_grad_list = [
                pserver_program.global_block().create_var(
                    name="%s.trainer_%d.pserver_%d" %
                    (table_grad_var.name, index, pserver_index),
                    type=table_grad_var.type,
                    shape=table_grad_var.shape,
                    dtype=table_grad_var.dtype)
                for index in range(self.trainer_num)
            ]

            # append sum op for pserver_side_table_grad_list
            table_opt_block.append_op(
                type="sum",
                inputs={"X": pserver_side_table_grad_list},
                outputs={"Out": [grad_var]})
        else:
            # in async_mode, for table gradient, it also need to be splited to each parameter server
            origin_grad_name = grad_var.name
            splited_grad_name = self.trainer_side_table_grad_list[
                pserver_index].name
            if not splited_grad_name.startswith(origin_grad_name):
                raise ValueError("origin_grad_var: " + splited_grad_name +
                                 " grad_var:" + grad_var.name)
            grad_var = pserver_program.global_block().rename_var(
                origin_grad_name, splited_grad_name)

        lr_var = pserver_program.global_block().vars[table_opt_op.input(
            "LearningRate")[0]]
        inputs = {
            "Param": [param_var],
            "Grad": [grad_var],
            "LearningRate": [lr_var]
        }
        outputs = {"ParamOut": [param_var]}
        table_opt_block.append_op(
            type=table_opt_op.type,
            inputs=inputs,
            outputs=outputs,
            attrs=table_opt_op.attrs)

        # add table parameter gradient and it's block id to grad_to_block_id
        grad_to_block_id.append(grad_var.name + ":" + str(table_opt_block.idx))

        return table_opt_block

    def _create_vars_from_blocklist(self,
                                    program,
                                    block_list,
                                    add_trainer_suffix=False):
        """
        Create vars for each split.
        NOTE: only grads need to be named for different trainers, use
              add_trainer_suffix to rename the grad vars.
        Args:
            program (ProgramDesc): ProgramDesc which gradients blong.
            block_list (list[(varname, block_id, block_size)]): List of gradient blocks.
            add_trainer_suffix (Bool): Add trainer suffix to new variable's name if set True.
        Returns: 
            var_mapping (dict(varname->[new_varname_variable])):A dict mapping 
                from original var name to each var split.
        """

        # varname->[(block_id, current_block_size)]
        block_map = dict()

        var_mapping = dict()
        for block_str in block_list:
            varname, offset, size = block_str.split(":")
            if not block_map.has_key(varname):
                block_map[varname] = []
            block_map[varname].append((long(offset), long(size)))
        # Do not remove this important debug message:
        print("block map: %s" % block_map)

        for varname, splited in block_map.iteritems():
            orig_var = program.global_block().var(varname)
            if len(splited) == 1:
                if self.sync_mode and add_trainer_suffix:
                    new_var_name = "%s.trainer_%d" % \
                        (orig_var.name, self.trainer_id)
                    program.global_block().rename_var(varname, new_var_name)
                    var_mapping[varname] = \
                        [program.global_block().var(new_var_name)]
                else:
                    var_mapping[varname] = \
                        [program.global_block().var(orig_var.name)]
                continue

            var_mapping[varname] = []
            orig_shape = orig_var.shape
            orig_dim1_flatten = 1
            if len(orig_shape) >= 2:
                orig_dim1_flatten = reduce(lambda x, y: x * y, orig_shape[1:])

            for i, block in enumerate(splited):
                size = block[1]
                rows = size / orig_dim1_flatten
                splited_shape = [rows]
                if len(orig_shape) >= 2:
                    splited_shape.extend(orig_shape[1:])
                new_var_name = ""
                if self.sync_mode and add_trainer_suffix:
                    new_var_name = "%s.block%d.trainer_%d" % \
                        (varname, i, self.trainer_id)
                else:
                    new_var_name = "%s.block%d" % \
                        (varname, i)
                var = program.global_block().create_var(
                    name=new_var_name,
                    persistable=False,
                    dtype=orig_var.dtype,
                    type=orig_var.type,
                    shape=splited_shape)  # flattend splited var
                var_mapping[varname].append(var)
            program.global_block().sync_with_cpp()
        return var_mapping

    def create_splited_vars(self, source_var, block, tag):
        return [
            block.create_var(
                name=str(source_var.name + tag + str(index)),
                type=source_var.type,
                shape=source_var.shape,
                dtype=source_var.dtype)
            for index in range(len(self.pserver_endpoints))
        ]

    def _clone_var(self, block, var, persistable=True):
        assert isinstance(var, Variable)
        return block.create_var(
            name=var.name,
            shape=var.shape,
            dtype=var.dtype,
            type=var.type,
            lod_level=var.lod_level,
            persistable=persistable)

    def _insert_split_op(self, program, orig_var, index, splited_vars):
        if orig_var.type == core.VarDesc.VarType.SELECTED_ROWS:
            height_sections = []
            for v in splited_vars:
                height_sections.append(v.shape[0])
            program.global_block().insert_op(
                index=index + 1,
                type="split_selected_rows",
                inputs={"X": orig_var},
                outputs={"Out": splited_vars},
                attrs={"height_sections": height_sections})
        elif orig_var.type == core.VarDesc.VarType.LOD_TENSOR:
            sections = []
            for v in splited_vars:
                sections.append(v.shape[0])
            program.global_block().insert_op(
                index=index + 1,
                type="split_byref",
                inputs={"X": orig_var},
                outputs={"Out": splited_vars},
                attrs={"sections": sections}  # assume split evenly
            )
        else:
            AssertionError("Variable type should be in set "
                           "[LOD_TENSOR, SELECTED_ROWS]")

    def _get_optimizer_input_shape(self, op_type, varkey, orig_shape,
                                   param_shape):
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
        elif op_type == "momentum":
            if varkey == "Velocity":
                return param_shape
        elif op_type == "":
            if varkey == "Moment":
                return param_shape
        elif op_type == "sgd":
            pass
        return orig_shape

    def _get_varname_parts(self, varname):
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

    def _orig_varname(self, varname):
        orig, _, _ = self._get_varname_parts(varname)
        return orig

    def _append_pserver_grad_merge_ops(self, optimize_block,
                                       grad_varname_for_block, endpoint,
                                       grad_to_block_id, origin_program):
        program = optimize_block.program
        pserver_block = program.global_block()
        grad_block = None
        for g in self.param_grad_ep_mapping[endpoint]["grads"]:
            if self._orig_varname(g.name) == \
                    self._orig_varname(grad_varname_for_block):
                grad_block = g
                break
        if not grad_block:
            # do not append this op if current endpoint
            # is not dealing with this grad block
            return
        orig_varname, block_name, trainer_name = self._get_varname_parts(
            grad_block.name)
        if block_name:
            merged_var_name = '.'.join([orig_varname, block_name])
        else:
            merged_var_name = orig_varname
        merged_var = \
            pserver_block.vars[merged_var_name]
        grad_to_block_id.append(merged_var.name + ":" + str(optimize_block.idx))
        if self.sync_mode and self.trainer_num > 1:
            vars2merge = []
            for i in xrange(self.trainer_num):
                per_trainer_name = "%s.trainer_%d" % \
                (merged_var_name, i)
                vars2merge.append(pserver_block.vars[per_trainer_name])

            optimize_block.append_op(
                type="sum",
                inputs={"X": vars2merge},
                outputs={"Out": merged_var})
            # TODO(panyx0718): What if it's SELECTED_ROWS.
            if not merged_var.type == core.VarDesc.VarType.SELECTED_ROWS:
                optimize_block.append_op(
                    type="scale",
                    inputs={"X": merged_var},
                    outputs={"Out": merged_var},
                    attrs={"scale": 1.0 / float(self.trainer_num)})
        return merged_var

    def _append_pserver_ops(self, optimize_block, opt_op, endpoint,
                            grad_to_block_id, origin_program, merged_var):
        program = optimize_block.program
        pserver_block = program.global_block()
        new_inputs = dict()
        # update param/grad shape first, then other inputs like
        # moment can use the updated shape
        for key in opt_op.input_names:
            if key == "Grad":
                new_inputs[key] = merged_var
            elif key == "Param":
                # param is already created on global program
                param_block = None
                for p in self.param_grad_ep_mapping[endpoint]["params"]:
                    if same_or_split_var(p.name, opt_op.input(key)[0]):
                        param_block = p
                        break
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
                if pserver_block.vars.has_key(lr_varname):
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
            if key in ["Param", "Grad", "LearningRate"]:
                continue
            var = self.origin_program.global_block().vars[opt_op.input(key)[0]]
            # update accumulator variable shape
            param_shape = new_inputs["Param"].shape
            new_shape = self._get_optimizer_input_shape(opt_op.type, key,
                                                        var.shape, param_shape)
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
            attrs=opt_op.attrs)

    def _is_splited_grad_var(self, var, var_dict):
        grad_block = None
        for _, g in var_dict.iteritems():
            if self._orig_varname(g.name) == self._orig_varname(var.name):
                if g.name.find(".trainer_") == -1:
                    grad_block = g
                    break
        return grad_block

    def _append_pserver_non_opt_ops(self, optimize_block, opt_op, endpoint):
        program = optimize_block.program
        # Append the ops for parameters that do not need to be optimized/updated
        inputs = self._get_input_map_from_op(
            self.origin_program.global_block().vars, opt_op)
        for key, varlist in inputs.iteritems():
            if not isinstance(varlist, list):
                varlist = [varlist]
            for var in varlist:
                # for ops like clipping and weight decay, get the splited var
                # for inputs/outputs
                grad_block = self._is_splited_grad_var(
                    var, program.global_block().vars)
                if grad_block:
                    inputs[key] = grad_block
                elif not program.global_block().vars.has_key(var.name):
                    program.global_block().create_var(
                        name=var.name,
                        persistable=var.persistable,
                        dtype=var.dtype,
                        shape=var.shape)

        outputs = self._get_output_map_from_op(
            self.origin_program.global_block().vars, opt_op)
        for key, varlist in outputs.iteritems():
            if not isinstance(varlist, list):
                varlist = [varlist]
            for var in varlist:
                grad_block = self._is_splited_grad_var(
                    var, program.global_block().vars)
                if grad_block:
                    outputs[key] = grad_block
                elif not program.global_block().vars.has_key(var.name):
                    program.global_block().clone_variable(var)

        optimize_block.append_op(
            type=opt_op.type,
            inputs=inputs,
            outputs=outputs,
            attrs=opt_op.attrs)

    def _is_op_connected(self, op1, op2):
        # If one op's input is another op's output or
        # one op's output is another op's input, we say
        # the two operator is connected.
        def _append_inname_remove_beta(varname_list):
            op_input_names = []
            for in_name in varname_list:
                # HACK: remove beta1 and beta2 to avoid let all
                # ops connected.
                if in_name.startswith("beta2_pow_acc") or \
                    in_name.startswith("beta1_pow_acc"):
                    continue
                else:
                    op_input_names.append(in_name)
            return op_input_names

        op1_input_names = _append_inname_remove_beta(op1.desc.input_arg_names())
        op1_output_names = op1.desc.output_arg_names()

        op2_input_names = _append_inname_remove_beta(op2.desc.input_arg_names())
        op2_output_names = op2.desc.output_arg_names()

        if set(op1_output_names) & set(op2_input_names) or \
           set(op1_input_names) & set(op2_output_names):
            return True
        return False

    def _create_ufind(self, optimize_ops):
        # Create a unit find data struct by optimize ops
        ufind = UnionFind(optimize_ops)
        for i in xrange(len(optimize_ops)):
            for j in xrange(i, len(optimize_ops)):
                op1 = optimize_ops[i]
                op2 = optimize_ops[j]
                if self._is_op_connected(op1, op2):
                    ufind.union(op1, op2)
        return ufind

    def _is_opt_role_op(self, op):
        # NOTE: depend on oprole to find out whether this op is for
        # optimize
        op_maker = core.op_proto_and_checker_maker
        optimize_role = core.op_proto_and_checker_maker.OpRole.Optimize
        if op_maker.kOpRoleAttrName() in op.attrs and \
            int(op.attrs[op_maker.kOpRoleAttrName()]) == int(optimize_role):
            return True
        return False

    def _is_optimizer_op(self, op):
        if "Param" in op.input_names and \
            "LearningRate" in op.input_names:
            return True
        return False

    def _is_opt_op_on_pserver(self, endpoint, op):
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

    def _get_input_map_from_op(self, varmap, op):
        """Returns a dict from op input name to the vars in varmap."""
        iomap = dict()
        for key in op.input_names:
            vars = []
            for varname in op.input(key):
                vars.append(varmap[varname])
            if len(vars) == 1:
                iomap[key] = vars[0]
            else:
                iomap[key] = vars
        return iomap

    def _get_output_map_from_op(self, varmap, op):
        """Returns a dict from op output name to the vars in varmap."""
        iomap = dict()
        for key in op.output_names:
            vars = []
            for varname in op.output(key):
                vars.append(varmap[varname])
            if len(vars) == 1:
                iomap[key] = vars[0]
            else:
                iomap[key] = vars
        return iomap

    def _get_lr_ops(self):
        lr_ops = []
        # find learning rate variables by optimize op
        lr_vars = set()
        for op in self.optimize_ops:
            if self._is_optimizer_op(op):
                lr_vars.add(op.input("LearningRate")[0])

        find_ops = []
        # find ops which output is lr var
        block = self.origin_program.global_block()
        for op in block.ops:
            if set(op.output_arg_names) & lr_vars:
                find_ops.append(op)
        # make a union find struct by the ops in default_main_program
        ufind = UnionFind(block.ops)

        for op1 in block.ops:
            for op2 in block.ops:
                # NOTE: we need to skip all optimize ops, since it is connected
                # with forward/backward ops and lr ops, we only need the lr ops.
                if op1 != op2 and self._is_op_connected(op1, op2) and \
                    not self._is_optimizer_op(op1) and not self._is_optimizer_op(op2):
                    ufind.union(op1, op2)
        # find all ops which is related with lr var
        for op1 in block.ops:
            for op2 in find_ops:
                if ufind.is_connected(op1, op2):
                    lr_ops.append(op1)
                    # we only need to append op for once
                    break
        return lr_ops

    def _get_optimize_pass(self):
        """
        Get optimizer operators, paramters and gradients from origin_program
        Returns:
            opt_ops (list): optimize operators.
            params_grads (dict): paramter->gradient.
        """
        block = self.origin_program.global_block()
        opt_ops = []
        params_grads = []
        origin_var_dict = self.origin_program.global_block().vars
        for op in block.ops:
            if self._is_opt_role_op(op):
                opt_ops.append(op)
                # HACK(wuyi): if we find grad vars from input of optimize
                # ops, we may get the output of clip op. Use syntax "@GRAD"
                # and op_role_var to get the pair.
                for input_name in op.input_arg_names:
                    if input_name.find("@GRAD") != -1 and \
                        op.attrs[RPC_OP_ROLE_ATTR_NAME]:
                        param_name = op.attrs[OP_ROLE_VAR_ATTR_NAME][0]
                        params_grads.append([
                            origin_var_dict[param_name],
                            origin_var_dict[input_name]
                        ])
            elif self._is_adam_connected_op(op):
                opt_ops.append(op)
            else:
                pass
        return opt_ops, params_grads

    def _is_adam_connected_op(self, op):
        """
        A hack function to determinate whether the input operator
        is connected to optimize operator.
        """
        if op.type == "scale":
            for in_name in op.input_arg_names:
                if in_name.startswith("beta1_pow_acc") or \
                        in_name.startswith("beta2_pow_acc"):
                    return True
        return False
