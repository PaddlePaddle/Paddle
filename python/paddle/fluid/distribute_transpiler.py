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
import framework
from framework import Program, default_main_program, default_startup_program, Parameter, Variable
import optimizer
from layer_helper import LayerHelper
from distributed_spliter import *
import math
from . import core


class VarBlock:
    def __init__(self, varname, offset, size):
        self.varname = varname
        # NOTE: real offset is offset * size
        self.offset = offset
        self.size = size

    def __str__(self):
        return "%s:%d:%d" % (self.varname, self.offset, self.size)


class UnionFind(object):
    """ Union-find data struct.
    
    Union-find is a data struct that keeps track of a set of elements partitioned
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


class DistributeTranspiler:
    def transpile(self,
                  optimize_ops,
                  params_grads,
                  trainer_id,
                  program=None,
                  pservers="127.0.0.1:6174",
                  trainers=1,
                  split_method=round_robin):
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

            :param optimize_ops: op list of optimization, should be the
                                    return value of Optimizer.minimize
            :type optimize_ops: list
            :param params_grads: list of tuple(weight, gradient)
            :type params_grads: list
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
        """
        assert (callable(split_method))
        if program is None:
            program = default_main_program()
        self.program = program
        self.trainers = trainers
        self.optimize_ops = optimize_ops
        # TODO(typhoonzero): currently trainer_id is fetched from cluster system
        # like Kubernetes, we should port this to use etcd later when developing
        # fluid distributed training with fault-tolerance.
        self.trainer_id = trainer_id
        pserver_endpoints = pservers.split(",")

        # step1
        param_list = [pg[0] for pg in params_grads]
        grad_list = [pg[1] for pg in params_grads]
        grad_blocks = split_dense_variable(grad_list, len(pserver_endpoints))
        param_blocks = split_dense_variable(param_list, len(pserver_endpoints))
        # step2
        grad_var_mapping = self._append_split_op(program, grad_blocks)
        # step3
        send_inputs = []
        send_outputs = []
        for b in grad_blocks:  # append by order
            varname, block_id, _ = b.split(":")
            send_inputs.append(grad_var_mapping[varname][int(block_id)])

        param_var_mapping = self._create_vars_from_blocklist(program,
                                                             param_blocks)
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
            name="RPC_CLIENT_VAR",
            persistable=True,
            type=core.VarDesc.VarType.RAW)

        # create send_op
        program.global_block().append_op(
            type="send",
            inputs={"X": send_inputs},
            outputs={"Out": send_outputs,
                     "RPCClient": rpc_client_var},
            attrs={"endpoints": pserver_endpoints,
                   "epmap": eplist})
        # step4
        for varname, splited_var in param_var_mapping.iteritems():
            if len(splited_var) <= 1:
                continue
            orig_param = program.global_block().vars[varname]
            program.global_block().append_op(
                type="concat",
                inputs={"X": splited_var},
                outputs={"Out": [orig_param]},
                attrs={"axis": 0})

    def get_trainer_program(self):
        # remove optimize ops and add a send op to main_program
        self.program.global_block().delete_ops(self.optimize_ops)
        # FIXME(typhoonzero): serialize once will fix error occurs when clone.
        self.program.__str__()
        return self.program

    def get_pserver_program(self, endpoint):
        """
        Get pserver side program using the endpoint.
        NOTE: assume blocks of the same variable is not distributed
        on the same pserver, only change param/grad varnames for
        trainers to fetch.
        """
        # step1
        pserver_program = Program()
        # step2
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
            pserver_program.global_block().create_var(
                name=orig_var_name,
                persistable=True,
                type=v.type,
                dtype=v.dtype,
                shape=v.shape)
            for trainer_id in xrange(self.trainers):
                var = pserver_program.global_block().create_var(
                    name="%s.trainer_%d" % (orig_var_name, trainer_id),
                    persistable=False,
                    type=v.type,
                    dtype=v.dtype,
                    shape=v.shape)
                recv_inputs.append(var)
        # step3
        optimize_block = pserver_program.create_block(0)
        # step 4
        # Create a union-find data struct from optimize ops,
        # If two ops are connected, we could add these two ops
        # into one set.
        ufind = self._create_ufind(self.optimize_ops)
        # step 4.2 
        # Iterate through the ops and append optimize op which
        # located on current pserver
        opt_op_on_pserver = []
        for _, op in enumerate(self.optimize_ops):
            if self._is_opt_op(op) and self._is_opt_op_on_pserver(endpoint, op):
                opt_op_on_pserver.append(op)
        # step 4.3
        # Iterate through the ops, and if an op and the optimize ops
        # which located on current pserver are in one set, then 
        # append it into the sub program.
        for _, op in enumerate(self.optimize_ops):
            for _, opt_op in enumerate(opt_op_on_pserver):
                if ufind.is_connected(op, opt_op):
                    if self._is_opt_op(op):
                        self._append_pserver_ops(optimize_block, op, endpoint,
                                                 default_main_program())
                    else:
                        self._append_pserver_non_opt_ops(optimize_block, op)
                    break
        # step5 append the listen_and_serv op
        pserver_program.global_block().append_op(
            type="listen_and_serv",
            inputs={'X': recv_inputs},
            outputs={},
            attrs={
                "OptimizeBlock": optimize_block,
                "endpoint": endpoint,
                "Fanin": self.trainers
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
        orig_s_prog = framework.default_startup_program()
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
            tmpvar = s_prog.global_block().create_var(
                name=var.name,
                persistable=var.persistable,
                dtype=var.dtype,
                shape=var.shape)
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
    def _create_vars_from_blocklist(self,
                                    program,
                                    block_list,
                                    add_trainer_suffix=False):
        """
        NOTE: only grads need to be named for different trainers, use
              add_trainer_suffix to rename the grad vars.
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
                if add_trainer_suffix:
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
                if add_trainer_suffix:
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

    def _clone_var(self, block, var):
        assert isinstance(var, Variable)
        return block.create_var(
            name=var.name,
            shape=var.shape,
            dtype=var.dtype,
            type=var.type,
            lod_level=var.lod_level,
            persistable=True)

    def _append_split_op(self, program, gradblocks):
        # Split variables that need to be split and append respective ops
        var_mapping = self._create_vars_from_blocklist(
            program, gradblocks, add_trainer_suffix=True)
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
                    type="split",
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
        return orig_var_name

    def _append_pserver_ops(self, optimize_block, opt_op, endpoint,
                            origin_program):
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
                            self._orig_varname(g.name), opt_op.input(key)[0]):
                        grad_block = g
                        break
                if not grad_block:
                    # do not append this op if current endpoint
                    # is not dealing with this grad block
                    return
                merged_var = \
                    pserver_block.vars[self._orig_varname(grad_block.name)]
                if self.trainers > 1:
                    vars2merge = []
                    for i in xrange(self.trainers):
                        per_trainer_name = "%s.trainer_%d" % \
                        (self._orig_varname(grad_block.name), i)
                        vars2merge.append(pserver_block.vars[per_trainer_name])

                    optimize_block.append_op(
                        type="sum",
                        inputs={"X": vars2merge},
                        outputs={"Out": merged_var})
                    if not merged_var.type == core.VarDesc.VarType.SELECTED_ROWS:
                        optimize_block.append_op(
                            type="scale",
                            inputs={"X": merged_var},
                            outputs={"Out": merged_var},
                            attrs={"scale": 1.0 / float(self.trainers)})
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
                # leraning rate variable has already be created by non-optimize op,
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
            var = self.program.global_block().vars[opt_op.input(key)[0]]
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
        outputs = self._get_output_map_from_op(self.program.global_block().vars,
                                               opt_op)
        outputs["ParamOut"] = new_inputs["Param"]

        optimize_block.append_op(
            type=opt_op.type,
            inputs=new_inputs,
            outputs=outputs,
            attrs=opt_op.attrs)

    def _append_pserver_non_opt_ops(self, optimize_block, opt_op):
        program = optimize_block.program
        # Append the ops for parameters that do not need to be optimized/updated
        inputs = self._get_input_map_from_op(self.program.global_block().vars,
                                             opt_op)
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

        outputs = self._get_output_map_from_op(self.program.global_block().vars,
                                               opt_op)

        for varlist in outputs.itervalues():
            if not isinstance(varlist, list):
                varlist = [varlist]

            for var in varlist:
                program.global_block().create_var(
                    name=var.name,
                    persistable=var.persistable,
                    dtype=var.dtype,
                    shape=var.shape)

        optimize_block.append_op(
            type=opt_op.type,
            inputs=inputs,
            outputs=outputs,
            attrs=opt_op.attrs)

    def _is_op_connected(self, op1, op2):
        # If one op's input is another op's output or
        # one op's output is another op's input, we say
        # the two operator is connected.
        op1_input_names = op1.desc.input_arg_names()
        op1_output_names = op1.desc.output_arg_names()

        op2_input_names = op2.desc.input_arg_names()
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
        if op.input("Param") in param_names:
            return True
        else:
            for n in param_names:
                param = op.input("Param")[0]
                if same_or_split_var(n, param) and n != param:
                    return True
            return False
        return False

    def _get_input_map_from_op(self, varmap, op):
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
