#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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
from framework import Program, default_main_program, Parameter, Variable
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

            :param optimize_ops: op list of optimization, should be the
                                 return value of Optimizer.minimize
            :type optimize_ops: list
            :param program: program to optimize, default is default_main_program
            :param pservers: parameter server endpoints like "m1:6174,m2:6174"
            :type pservers: string
            :return: return a list of programs
        """
        assert (callable(split_method))
        if program is None:
            program = default_main_program()
        self.program = program
        self.trainers = trainers
        self.optimize_ops = optimize_ops
        # steps to transpile:
        # 1. split variable to multiple blocks, aligned by product(dim[1:]) (width).
        # 2. modify trainer program add split_op to each Grad.
        # 3. append send_op to trainer.
        # 4. append concat_op to trainer to update local weights.
        # 5. create new program for parameter server.
        # 6. create parameter server program by split_method generated endpoint->VarBlock

        pserver_endpoints = pservers.split(",")

        # step1
        param_list = [pg[0] for pg in params_grads]
        grad_list = [pg[1] for pg in params_grads]
        # TODO: add split selected rows support
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
            psersistable=True,
            dtype='float32',  # dtype and shape is not used in fact
            shape=[0])

        # create send_op
        send_op = program.global_block().append_op(
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
            concat = program.global_block().append_op(
                type="concat",
                inputs={"X": splited_var},
                outputs={"Out": [orig_param]},
                attrs={"axis": 0})

    def _create_vars_from_blocklist(self, program, block_list):
        # Create respective variables using the block_list
        block_map = dict()
        var_mapping = dict()
        for block_str in block_list:
            varname, offset, size = block_str.split(":")
            if not block_map.has_key(varname):
                block_map[varname] = []
            block_map[varname].append((long(offset), long(size)))
        for varname, splited in block_map.iteritems():
            orig_var = program.global_block().vars[varname]
            var_mapping[varname] = []
            if len(splited) == 1:
                var_mapping[varname] = [orig_var]
                continue
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
                var = program.global_block().create_var(
                    name="%s.block%d" % (varname, i),
                    psersistable=False,
                    dtype=orig_var.dtype,
                    shape=splited_shape)  # flattend splited var
                var_mapping[varname].append(var)
        return var_mapping

    def _clone_var(self, block, var):
        assert isinstance(var, Variable)
        return block.create_var(
            name=var.name,
            shape=var.shape,
            dtype=var.dtype,
            type=var.type,
            lod_level=var.lod_level,
            # HACK: let all param in pserver be persistable so the child
            # program in recv can get them
            persistable=True)

    def _append_split_op(self, program, gradblocks):
        # Split variables that need to be split and append respective ops
        var_mapping = self._create_vars_from_blocklist(program, gradblocks)
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

    def get_trainer_program(self):
        # remove optimize ops and add a send op to main_program
        self.program.global_block().delete_ops(self.optimize_ops)
        return self.program

    def _create_var_for_trainers(self, block, var, trainers):
        # For each trainer, create the necessary variables
        var_list = []
        for i in xrange(trainers):
            var_each = block.create_var(
                name="%s.trainer_%d" % (var.name, i),
                psersistable=var.persistable,
                dtype=var.dtype,
                shape=var.shape)
            var_list.append(var_each)
        return var_list

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

    def _is_op_on_pserver(self, endpoint, all_ops, idx):
        """
        Recursively check if the op need to run on current server.
        Assume that ops are in the execution order.
        """
        param_names = [
            p.name for p in self.param_grad_ep_mapping[endpoint]["params"]
        ]
        op = all_ops[idx]
        if op.inputs.has_key("Param"):
            if op.inputs["Param"].name in param_names:
                return True
            else:
                for n in param_names:
                    if same_or_split_var(n, op.inputs[
                            "Param"].name) and n != op.inputs["Param"].name:
                        return True
                return False
        else:
            j = idx - 1
            while j >= 0:
                prev_op = all_ops[j]
                prev_output_names = [o.name for o in prev_op.outputs.values()]
                prev_input_names = [o.name for o in prev_op.inputs.values()]
                found1 = False
                found2 = False
                for _, v in op.inputs.iteritems():
                    if v.name in prev_output_names:
                        found1 = self._is_op_on_pserver(endpoint, all_ops, j)
                # later ops may produce output for prev op's next batch use.
                for _, v in op.outputs.iteritems():
                    if v.name in prev_input_names:
                        found2 = self._is_op_on_pserver(endpoint, all_ops, j)
                if found1 or found2:
                    return True
                j -= 1
            return False

    def _append_pserver_ops(self, program, pserver_program, opt_op, endpoint):
        new_inputs = dict()
        # update param/grad shape first, then other inputs like
        # moment can use the updated shape
        for key, var in opt_op.inputs.iteritems():
            if key == "Grad":
                grad_block = None
                for g in self.param_grad_ep_mapping[endpoint]["grads"]:
                    if same_or_split_var(g.name, var.name):
                        grad_block = g
                        break
                if not grad_block:
                    # do not append this op if current endpoint
                    # is not dealing with this grad block
                    return
                merged_var = program.global_block().create_var(
                    name=grad_block.name,
                    persistable=grad_block.persistable,
                    dtype=grad_block.dtype,
                    shape=grad_block.shape)
                # append merging ops if trainers > 1
                if self.trainers > 1:
                    vars2merge = self._create_var_for_trainers(
                        program.global_block(), grad_block, self.trainers)
                    program.global_block().append_op(
                        type="sum",
                        inputs={"X": vars2merge},
                        outputs={"Out": merged_var})
                    program.global_block().append_op(
                        type="scale",
                        inputs={"X": merged_var},
                        outputs={"Out": merged_var},
                        attrs={"scale": 1.0 / float(self.trainers)})
                new_inputs[key] = merged_var
            elif key == "Param":
                # param is already created on global program
                param_block = None
                for p in self.param_grad_ep_mapping[endpoint]["params"]:
                    if same_or_split_var(p.name, var.name):
                        param_block = p
                        break
                if not param_block:
                    return
                tmpvar = program.global_block().create_var(
                    name=param_block.name,
                    persistable=True,
                    dtype=param_block.dtype,
                    shape=param_block.shape)

                new_inputs[key] = tmpvar

        for key, var in opt_op.inputs.iteritems():
            if key in ["Param", "Grad"]:
                continue
            # update accumulator variable shape
            param_shape = new_inputs["Param"].shape
            new_shape = self._get_optimizer_input_shape(opt_op.type, key,
                                                        var.shape, param_shape)
            tmpvar = program.global_block().create_var(
                name=var.name,
                persistable=var.persistable,
                dtype=var.dtype,
                shape=new_shape)
            new_inputs[key] = tmpvar
            # create var in pserver program global block.
            # TODO(typhoonzero): put blocks in one program to avoid create two
            # variables.
            pserver_program.global_block().create_var(
                name=var.name,
                persistable=var.persistable,
                dtype=var.dtype,
                shape=new_shape)

        # change output's ParamOut variable
        opt_op.outputs["ParamOut"] = new_inputs["Param"]
        program.global_block().append_op(
            type=opt_op.type,
            inputs=new_inputs,
            outputs=opt_op.outputs,
            attrs=opt_op.attrs)

    def _append_pserver_non_opt_ops(self, program, pserver_program, opt_op):
        # Append the ops for parameters that do not need to be optimized/updated
        for _, var in opt_op.inputs.iteritems():
            program.global_block().create_var(
                name=var.name,
                persistable=var.persistable,
                dtype=var.dtype,
                shape=var.shape)
            pserver_program.global_block().create_var(
                name=var.name,
                persistable=var.persistable,
                dtype=var.dtype,
                shape=var.shape)
        program.global_block().append_op(
            type=opt_op.type,
            inputs=opt_op.inputs,
            outputs=opt_op.outputs,
            attrs=opt_op.attrs)

    def get_pserver_program(self, endpoint):
        """
        Get pserver side program using the endpoint

        NOTE: assume blocks of the same variable is not distributed
        on the same pserver, only change param/grad varnames for
        trainers to fetch. For each pserver endpoint, server side
        program must be a sub-set of the original optimization program.
        """
        # step5
        pserver_program = Program()
        for v in self.param_grad_ep_mapping[endpoint]["params"]:
            self._clone_var(pserver_program.global_block(), v)
        for v in self.param_grad_ep_mapping[endpoint]["grads"]:
            # create vars for each trainer in global scope, so
            # we don't need to create them when grad arrives.
            pserver_program.global_block().create_var(
                name=v.name, persistable=True, dtype=v.dtype, shape=v.shape)
            for trainer_id in xrange(self.trainers):
                print("create variable for program: %s.trainer_%d" %
                      (v.name, trainer_id))
                pserver_program.global_block().create_var(
                    name="%s.trainer_%d" % (v.name, trainer_id),
                    persistable=True,
                    dtype=v.dtype,
                    shape=v.shape)
        # step6
        optimize_sub_program = Program()
        # Iterate through the ops and append ops as needed
        for idx, opt_op in enumerate(self.optimize_ops):
            is_op_on_pserver = self._is_op_on_pserver(endpoint,
                                                      self.optimize_ops, idx)
            if not is_op_on_pserver:
                continue
            if opt_op.inputs.has_key("Grad"):
                self._append_pserver_ops(optimize_sub_program, pserver_program,
                                         opt_op, endpoint)
            else:
                self._append_pserver_non_opt_ops(optimize_sub_program,
                                                 pserver_program, opt_op)
        # Append the listen_and_serv op
        pserver_program.global_block().append_op(
            type="listen_and_serv",
            inputs={},
            outputs={},
            attrs={
                "OptimizeBlock": optimize_sub_program.global_block(),
                "endpoint": endpoint,
                "ParamList": [
                    p.name
                    for p in self.param_grad_ep_mapping[endpoint]["params"]
                ],
                "GradList": [
                    p.name
                    for p in self.param_grad_ep_mapping[endpoint]["grads"]
                ],
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
            new_outputs = dict()
            # do not append startup op if var is not on this pserver
            op_on_pserver = False
            for key, var in op.outputs.iteritems():
                newname, _ = _get_splited_name_and_shape(var.name)
                if newname:
                    op_on_pserver = True
                    new_outputs[key] = created_var_map[newname]
                elif var.name in pserver_vars:
                    op_on_pserver = True
                    new_outputs[key] = pserver_vars[var.name]

            if op_on_pserver:
                if op.type in [
                        "gaussian_random", "fill_constant", "uniform_random"
                ]:
                    op.attrs["shape"] = new_outputs["Out"].shape
                s_prog.global_block().append_op(
                    type=op.type,
                    inputs=op.inputs,
                    outputs=new_outputs,
                    attrs=op.attrs)
        return s_prog
