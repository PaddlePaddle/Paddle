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

import math

import distributed_splitter as splitter
from .. import core
from ..framework import Program, default_main_program, \
                        default_startup_program, \
                        Variable, Parameter, grad_var_name

LOOKUP_TABLE_TYPE = "lookup_table"
LOOKUP_TABLE_GRAD_TYPE = "lookup_table_grad"
RPC_CLIENT_VAR_NAME = "RPC_CLIENT_VAR"


class VarBlock:
    def __init__(self, varname, offset, size):
        self.varname = varname
        # NOTE: real offset is offset * size
        self.offset = offset
        self.size = size

    def __str__(self):
        return "%s:%d:%d" % (self.varname, self.offset, self.size)


class UnionFind(object):
    """ Union-find data structure.

    Union-find is a data structure that keeps track of a set of elements partitioned
    into a number of disjoint (non-overlapping) subsets.

    Reference:
    https://en.wikipedia.org/wiki/Disjoint-set_data_structure

    Args:
      elements(list): The initialize element list.
    """

    def __init__(self, elementes=None):
        self._parents = []  # index -> parent index
        self._index = {}  # element -> index
        self._curr_idx = 0
        if not elementes:
            elementes = []
        for ele in elementes:
            self._parents.append(self._curr_idx)
            self._index.update({ele: self._curr_idx})
            self._curr_idx += 1

    def find(self, x):
        # Find the root index of given element x,
        # execute the path compress while findind the root index
        if not x in self._index:
            return -1
        idx = self._index[x]
        while idx != self._parents[idx]:
            t = self._parents[idx]
            self._parents[idx] = self._parents[t]
            idx = t
        return idx

    def union(self, x, y):
        # Union two given element
        x_root = self.find(x)
        y_root = self.find(y)

        if x_root == y_root:
            return
        self._parents[x_root] = y_root

    def is_connected(self, x, y):
        # If two given elements have the same root index,
        # then they are connected.
        return self.find(x) == self.find(y)


def same_or_split_var(p_name, var_name):
    return p_name == var_name or p_name.startswith(var_name + ".block")


def split_dense_variable(var_list,
                         pserver_count,
                         min_block_size=1024,
                         max_block_size=1048576):
    """
        We may need to split dense tensor to one or more blocks and put
        them equally onto parameter server. One block is a sub-tensor
        aligned by dim[0] of the tensor.

        We need to have a minimal block size so that the calculations in
        the parameter server side can gain better performance. By default
        minimum block size is 1024. The max block size is used to prevent
        very large blocks that may cause send error.
        :return: A list of VarBlocks. Each VarBlock specifies a shard of
           the var.
    """
    blocks = []
    for var in var_list:
        split_count = pserver_count
        var_numel = reduce(lambda x, y: x * y, var.shape)
        max_pserver_count = int(math.floor(var_numel / float(min_block_size)))
        if max_pserver_count == 0:
            max_pserver_count = 1
        if max_pserver_count < pserver_count:
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


def delete_ops(block, ops):
    try:
        start = list(block.ops).index(ops[0])
        end = list(block.ops).index(ops[-1])
        [block.remove_op(start) for _ in xrange(end - start + 1)]
    except Exception, e:
        raise e
    block.program.sync_with_cpp()


class DistributeTranspiler:
    def transpile(self,
                  trainer_id,
                  program=None,
                  pservers="127.0.0.1:6174",
                  trainers=1,
                  split_method=splitter.round_robin,
                  sync_mode=True):
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
        assert (callable(split_method))
        if program is None:
            program = default_main_program()
        self.origin_program = program
        self.trainer_num = trainers
        self.sync_mode = sync_mode
        # TODO(typhoonzero): currently trainer_id is fetched from cluster system
        # like Kubernetes, we should port this to use etcd later when developing
        # fluid distributed training with fault-tolerance.
        self.trainer_id = trainer_id
        pserver_endpoints = pservers.split(",")
        self.pserver_endpoints = pserver_endpoints
        self.optimize_ops, params_grads = self._get_optimize_pass()

        # process lookup_table_op
        # 1. check all lookup_table_op is distributed
        # 2. check all lookup_table_op share the same table.
        distributed_lookup_table_ops = []
        # support only one distributed_lookup_table now
        self.table_name = None
        for op in program.global_block().ops:
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

        self.has_distributed_lookup_table = len(
            distributed_lookup_table_ops) > 0

        # step1: For large parameters and gradients, split them into smaller
        # blocks.
        param_list = []
        grad_list = []
        for p, g in params_grads:
            # skip parameter marked not trainable
            if type(p) == Parameter and p.trainable == False:
                continue
            param_list.append(p)
            grad_list.append(g)

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
            self.table_grad_list = [
                program.global_block().create_var(
                    name="%s.trainer_%d.pserver_%d" %
                    (table_grad_var.name, trainer_id, index),
                    type=table_grad_var.type,
                    shape=table_grad_var.shape,
                    dtype=table_grad_var.dtype)
                for index in range(len(self.pserver_endpoints))
            ]

        grad_blocks = split_dense_variable(grad_list, len(pserver_endpoints))
        param_blocks = split_dense_variable(param_list, len(pserver_endpoints))
        # step2: Create new vars for the parameters and gradients blocks and
        # add ops to do the split.
        grad_var_mapping = self._append_split_op(program, grad_blocks)
        param_var_mapping = self._create_vars_from_blocklist(program,
                                                             param_blocks)
        # step3: Add gradients as send op inputs and parameters as send
        # op outputs.
        send_inputs = []
        send_outputs = []
        for b in grad_blocks:  # append by order
            varname, block_id, _ = b.split(":")
            send_inputs.append(grad_var_mapping[varname][int(block_id)])
        for b in param_blocks:
            varname, block_id, _ = b.split(":")
            send_outputs.append(param_var_mapping[varname][int(block_id)])
        # let send_op know which endpoint to send which var to, eplist has the same
        # order as send_inputs.
        eplist = split_method(send_inputs, pserver_endpoints)
        # create mapping of endpoint -> split var to create pserver side program
        self.param_grad_ep_mapping = dict()
        for i, ep in enumerate(eplist):
            param = send_outputs[i]
            grad = send_inputs[i]
            if not self.param_grad_ep_mapping.has_key(ep):
                self.param_grad_ep_mapping[ep] = {"params": [], "grads": []}
            self.param_grad_ep_mapping[ep]["params"].append(param)
            self.param_grad_ep_mapping[ep]["grads"].append(grad)

        rpc_client_var = program.global_block().create_var(
            name=RPC_CLIENT_VAR_NAME,
            persistable=True,
            type=core.VarDesc.VarType.RAW)

        # create send_op
        program.global_block().append_op(
            type="send",
            inputs={"X": send_inputs},
            outputs={"Out": send_outputs,
                     "RPCClient": rpc_client_var},
            attrs={
                "endpoints": pserver_endpoints,
                "epmap": eplist,
                "sync_mode": self.sync_mode
            })
        # step4: Concat the parameters splits together after recv.
        for varname, splited_var in param_var_mapping.iteritems():
            if len(splited_var) <= 1:
                continue
            orig_param = program.global_block().vars[varname]
            program.global_block().append_op(
                type="concat",
                inputs={"X": splited_var},
                outputs={"Out": [orig_param]},
                attrs={"axis": 0})

        if self.has_distributed_lookup_table:
            self._replace_lookup_table_op_with_prefetch(program, rpc_client_var,
                                                        eplist)
            self._split_table_grad_and_add_send_vars(program, rpc_client_var,
                                                     pserver_endpoints)

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
            if self._is_opt_op(op) and self._is_opt_op_on_pserver(endpoint, op):
                opt_op_on_pserver.append(op)
        # step 3.3
        # Iterate through the ops, and if an op and the optimize ops
        # which located on current pserver are in one set, then
        # append it into the sub program.

        # We try to put optimization program run parallelly, assume
        # optimization program always looks like:
        #
        # prevop -> prevop -> opt op -> following op -> following op; ->
        # prevop -> prevop -> opt op -> following op -> following op; ->
        # global op -> global op
        #
        # we put operators that can run parallelly to many program blocks.
        # in above example, we seperate ops by the ";". Global ops must run
        # after all the optimize ops finished.

        global_ops = []
        # HACK: optimization global ops only used to scale beta1 and beta2
        # replace it with dependency engine.
        for op in self.optimize_ops:
            if self._is_adam_connected_op(op):
                global_ops.append(op)

        def __append_optimize_op__(op, block, grad_to_block_id):
            if self._is_opt_op(op):
                self._append_pserver_ops(block, op, endpoint, grad_to_block_id,
                                         default_main_program())
            else:
                self._append_pserver_non_opt_ops(block, op)

        # append lr decay ops to the child block if exists
        lr_ops = self._get_lr_ops()
        if len(lr_ops) > 0:
            lr_decay_block = pserver_program.create_block(
                pserver_program.num_blocks - 1)
            for _, op in enumerate(lr_ops):
                self._append_pserver_non_opt_ops(lr_decay_block, op)

        # append op to the current block
        grad_to_block_id = []
        pre_block_idx = pserver_program.num_blocks - 1
        for idx, opt_op in enumerate(opt_op_on_pserver):
            per_opt_block = pserver_program.create_block(pre_block_idx)
            for _, op in enumerate(self.optimize_ops):
                # optimizer is connected to itself
                if ufind.is_connected(op, opt_op) and op not in global_ops:
                    __append_optimize_op__(op, per_opt_block, grad_to_block_id)

        # append global ops
        if global_ops:
            opt_state_block = pserver_program.create_block(
                pserver_program.num_blocks - 1)
            for glb_op in global_ops:
                __append_optimize_op__(glb_op, opt_state_block,
                                       grad_to_block_id)

        # NOT USED: single block version:
        #
        # for _, op in enumerate(self.optimize_ops):
        #     for _, opt_op in enumerate(opt_op_on_pserver):
        #         if ufind.is_connected(op, opt_op):
        #             __append_optimize_op__(glb_op, optimize_block)
        #             break

        # process distributed lookup_table
        prefetch_block = None
        if self.has_distributed_lookup_table:
            pserver_index = self.pserver_endpoints.index(endpoint)
            table_opt_block = self._create_table_optimize_block(
                pserver_index, pserver_program, pre_block_idx)
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
                "grad_to_block_id": grad_to_block_id,
                "Checkpoint": "/tmp/tangwei_ckpt/"
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

    # transpiler function for dis lookup_table
    def _replace_lookup_table_op_with_prefetch(self, program, rpc_client_var,
                                               eplist):
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
                        outputs={
                            "Out": self.prefetch_output_vars,
                            "RPCClient": rpc_client_var
                        },
                        attrs={"epmap": eplist})

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

    def _split_table_grad_and_add_send_vars(self, program, rpc_client_var,
                                            pserver_endpoints):
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
                    outputs={"Out": self.table_grad_list})
                program.global_block().insert_op(
                    index=op_index + 2,
                    type="send_vars",
                    inputs={'X': self.table_grad_list},
                    outputs={"RPCClient": rpc_client_var},
                    attrs={"sync_send": True,
                           "epmap": pserver_endpoints})
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
                                     pre_block_idx):
        def _clone_var(block, var, persistable=True):
            assert isinstance(var, Variable)
            return block.create_var(
                name=var.name,
                shape=var.shape,
                dtype=var.dtype,
                type=var.type,
                persistable=persistable)

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
        grad_var = _clone_var(
            pserver_program.global_block(),
            self.origin_program.global_block().vars[grad_var_name(
                self.table_name)],
            persistable=False)

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
            table_grad_list = [
                pserver_program.global_block().create_var(
                    name="%s.trainer_%d.pserver_%d" %
                    (table_grad_var.name, index, pserver_index),
                    type=table_grad_var.type,
                    shape=table_grad_var.shape,
                    dtype=table_grad_var.dtype)
                for index in range(self.trainer_num)
            ]

            # append sum op for table_grad_list
            table_opt_block.append_op(
                type="sum",
                inputs={"X": table_grad_list},
                outputs={"Out": [grad_var]})

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

        return table_opt_block

    # ====================== private transpiler functions =====================
    def _create_vars_from_blocklist(self,
                                    program,
                                    block_list,
                                    add_trainer_suffix=False):
        """
        Create vars for each split.
        NOTE: only grads need to be named for different trainers, use
              add_trainer_suffix to rename the grad vars.
        :return: A dict mapping from original var name to each var split.
        """
        block_map = dict()
        var_mapping = dict()
        for block_str in block_list:
            varname, offset, size = block_str.split(":")
            if not block_map.has_key(varname):
                block_map[varname] = []
            block_map[varname].append((long(offset), long(size)))
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

    def _append_split_op(self, program, gradblocks):
        # Split variables that need to be split and append respective ops
        add_suffix = False
        if self.trainer_num > 1:
            add_suffix = True
        var_mapping = self._create_vars_from_blocklist(
            program, gradblocks, add_trainer_suffix=add_suffix)
        for varname, splited_vars in var_mapping.iteritems():
            # variable that don't need to split have empty splited_vars
            if len(splited_vars) <= 1:
                continue
            orig_var = program.global_block().vars[varname]
            if orig_var.type == core.VarDesc.VarType.SELECTED_ROWS:
                height_sections = []
                for v in splited_vars:
                    height_sections.append(v.shape[0])
                program.global_block().append_op(
                    type="split_selected_rows",
                    inputs={"X": orig_var},
                    outputs={"Out": splited_vars},
                    attrs={"height_sections": height_sections})
            elif orig_var.type == core.VarDesc.VarType.LOD_TENSOR:
                sections = []
                for v in splited_vars:
                    sections.append(v.shape[0])
                program.global_block().append_op(
                    type="split_byref",
                    inputs={"X": orig_var},
                    outputs={"Out": splited_vars},
                    attrs={"sections": sections}  # assume split evenly
                )
            else:
                AssertionError("Variable type should be in set "
                               "[LOD_TENSOR, SELECTED_ROWS]")
        return var_mapping

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

    def _orig_varname(self, varname):
        suff_idx = varname.find(".trainer_")
        orig_var_name = ""
        if suff_idx >= 0:
            orig_var_name = varname[:suff_idx]
        else:
            orig_var_name = varname
        return orig_var_name

    def _append_pserver_ops(self, optimize_block, opt_op, endpoint,
                            grad_to_block_id, origin_program):
        program = optimize_block.program
        pserver_block = program.global_block()
        new_inputs = dict()
        # update param/grad shape first, then other inputs like
        # moment can use the updated shape
        for key in opt_op.input_names:
            if key == "Grad":
                grad_block = None
                for g in self.param_grad_ep_mapping[endpoint]["grads"]:
                    if same_or_split_var(
                            self._orig_varname(g.name),
                            self._orig_varname(opt_op.input(key)[0])):
                        grad_block = g
                        break
                if not grad_block:
                    # do not append this op if current endpoint
                    # is not dealing with this grad block
                    return
                merged_var = \
                    pserver_block.vars[self._orig_varname(grad_block.name)]
                grad_to_block_id.append(merged_var.name + ":" + str(
                    optimize_block.idx))
                if self.sync_mode and self.trainer_num > 1:
                    vars2merge = []
                    for i in xrange(self.trainer_num):
                        per_trainer_name = "%s.trainer_%d" % \
                        (self._orig_varname(grad_block.name), i)
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

    def _append_pserver_non_opt_ops(self, optimize_block, opt_op):
        program = optimize_block.program
        # Append the ops for parameters that do not need to be optimized/updated
        inputs = self._get_input_map_from_op(
            self.origin_program.global_block().vars, opt_op)
        for varlist in inputs.itervalues():
            if not isinstance(varlist, list):
                varlist = [varlist]

            for var in varlist:
                if not program.global_block().vars.has_key(var.name):
                    program.global_block().create_var(
                        name=var.name,
                        persistable=var.persistable,
                        dtype=var.dtype,
                        shape=var.shape)

        outputs = self._get_output_map_from_op(
            self.origin_program.global_block().vars, opt_op)

        for varlist in outputs.itervalues():
            if not isinstance(varlist, list):
                varlist = [varlist]

            for var in varlist:
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

    def _is_opt_op(self, op):
        # NOTE: It's a HACK implement.
        # optimize op: SGDOptimize, MomentumOptimizer, AdamOptimizer and etc...
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
            if self._is_opt_op(op):
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
                    not self._is_opt_op(op1) and not self._is_opt_op(op2):
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
        block = self.origin_program.global_block()
        opt_ops = []
        params_grads = []
        for op in block.ops:
            if self._is_opt_op(op):
                opt_ops.append(op)
                params_grads.append((self.origin_program.global_block().var(
                    op.input("Param")[0]),
                                     self.origin_program.global_block().var(
                                         op.input("Grad")[0])))
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
