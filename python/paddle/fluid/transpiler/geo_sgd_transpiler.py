#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import sys
import collections
import six
import numpy as np

from .ps_dispatcher import RoundRobin, PSDispatcher
from .. import core, framework
from ..framework import Program, default_main_program, \
    default_startup_program, Block, Parameter
from .details import wait_server_ready, VarsDistributed
from .details import delete_ops
from ..distribute_lookup_table import find_distributed_lookup_table
from .distribute_transpiler import DistributeTranspiler,DistributeTranspilerConfig,slice_variable,same_or_split_var

RPC_OP_ROLE_ATTR_NAME = op_role_attr_name = core.op_proto_and_checker_maker.kOpRoleAttrName()
RPC_OP_ROLE_ATTR_VALUE = core.op_proto_and_checker_maker.OpRole.RPC
PRINT_LOG = False

class GeoSgdTranspiler(DistributeTranspiler):
    def __init__(self, config=None):
        if config is not None:
            self.config = config
        else:
            self.config = DistributeTranspilerConfig()

        if self.config.split_method is None:
            self.config.split_method = RoundRobin

        global PRINT_LOG
        if self.config.print_log:
            PRINT_LOG = True

        assert (self.config.min_block_size >= 8192)
        assert (self.config.split_method.__bases__[0] == PSDispatcher)

    def transpile(self,
                  trainer_id,
                  program=None,
                  pservers="127.0.0.1:6174",
                  trainers=1,
                  sync_mode=False,
                  startup_program=None,
                  current_endpoint="127.0.0.1:6174"):
        """
        Run the transpiler. Transpile the input program.

        Args:
            trainer_id (int): id for current trainer worker, if you have
                n workers, the id may range from 0 ~ n-1
            program (Program|None): program to transpile,
                default is fluid.default_main_program().
            startup_program (Program|None): startup_program to transpile,
                default is fluid.default_startup_program().
            pservers (str): comma separated ip:port string for the pserver
                list.
            trainers (int|str): in pserver mode this is the number of
                trainers, in nccl2 mode this is a string of trainer
                endpoints.
            sync_mode (bool): Do sync training or not, default is True.
            startup_program (Program|None): startup_program to transpile,
                default is fluid.default_main_program().
            current_endpoint (str): need pass current endpoint when
                transpile as nccl2 distributed mode. In pserver mode
                this argument is not used.

        Examples:
            .. code-block:: python

                transpiler = fluid.DistributeTranspiler()
                t.transpile(
                    trainer_id=0,
                    pservers="127.0.0.1:7000,127.0.0.1:7001",
                    trainers=2,
                    sync_mode=False,
                    current_endpoint="127.0.0.1:7000")
        """
        if program is None:
            program = default_main_program()
        if startup_program is None:
            startup_program = default_startup_program()
        self.origin_program = program
        self.startup_program = startup_program
        self.origin_startup_program = self.startup_program.clone()

        self.trainer_num = trainers
        # geo-sgd only supply async-mode
        self.sync_mode = False
        self.trainer_id = trainer_id
        pserver_endpoints = pservers.split(",")
        self.pserver_endpoints = pserver_endpoints
        self.vars_overview = VarsDistributed()
        self.optimize_ops, self.params_grads = self._get_optimize_pass()

        ps_dispatcher = self.config.split_method(self.pserver_endpoints)
        self.table_name = find_distributed_lookup_table(self.origin_program)
        self.has_distributed_lookup_table = self.table_name != None
        self.param_name_to_grad_name = dict()
        self.grad_name_to_param_name = dict()
        for param_var, grad_var in self.params_grads:
            self.param_name_to_grad_name[param_var.name] = grad_var.name
            self.grad_name_to_param_name[grad_var.name] = param_var.name

        # get all sparse update ops
        self.sparse_update_ops = self._get_all_remote_sparse_update_op(
            self.origin_program)
        # use_sparse_update_param_name -> split_height_section
        self.sparse_param_to_height_sections = dict()

        # add distributed attrs to program
        self.origin_program._is_distributed = True
        self.origin_program._endpoints = self.pserver_endpoints
        self.origin_program._ps_endpoint = current_endpoint
        self.origin_program._is_chief = self.trainer_id == 0
        self.origin_program._distributed_lookup_table = self.table_name if self.table_name else None
        self.vars_info = collections.OrderedDict()
        self.split_to_origin_mapping = collections.OrderedDict()
        # split and create vars, then put splited vars in dicts for later use.
        # step 1. split and create vars, then put splited vars in dicts for later use.
        self._init_splited_vars()

        # step 3. create send var (param after optimize)
        send_vars = []
        ps_dispatcher.reset()
        param_var_mapping_items = list(six.iteritems(self.param_var_mapping))
        for param_varname, splited_vars in param_var_mapping_items:
            for _, var in enumerate(splited_vars):
                send_vars.append(var)

        recv_vars = send_vars
        ps_dispatcher.reset()
        eplist = ps_dispatcher.dispatch(recv_vars)
        for i, ep in enumerate(eplist):
            self.param_opt_ep_mapping[ep]["params"].append(recv_vars[i])
            distributed_var = self.vars_overview.get_distributed_var_by_slice(
                recv_vars[i].name)
            distributed_var.endpoint = ep
            origin_name = self.split_to_origin_mapping[recv_vars[i].name]
            self.vars_info[origin_name]["epmap"].append(ep)

        # batch training loop end flag
        dummy_output = program.global_block().create_var(
            name=framework.generate_control_dev_var_name())
        batch_num = program.global_block().create_var(
            name="batch_num")
        program.global_block().append_op(
            type="send",
            inputs={"X": [batch_num]},
            outputs={"Out": dummy_output},
            attrs={
                "send_varnames": [batch_num.name]}
        )

        self._get_trainer_startup_program(recv_vars=recv_vars, eplist=eplist)
        self.origin_program._parameters_on_pservers = self.vars_overview

    def _get_vars_info(self):
        # Todo :for dense param test,delete it when sparse method finish
        self.new_vars_info = collections.OrderedDict()
        sparse_str = 'SparseFeatFactors'
        for var in self.vars_info:
            if sparse_str not in var:
                self.new_vars_info[var] = self.vars_info[var]
        print(str(self.new_vars_info))
        return self.new_vars_info
            
    def _get_trainer_startup_program(self, recv_vars, eplist):
        startup_program = self.startup_program

        for varname, splited_var in six.iteritems(self.param_var_mapping):
            # Get the eplist of recv vars
            eps = []
            for var in splited_var:
                index = [v.name for v in recv_vars].index(var.name)
                eps.append(eplist[index])

            for var in splited_var:
                if startup_program.global_block().has_var(var.name):
                    continue
                startup_program.global_block().create_var(
                    name=var.name,
                    persistable=False,
                    type=var.type,
                    dtype=var.dtype,
                    shape=var.shape,
                    lod_level=var.lod_level)
            startup_program.global_block().append_op(
                type="recv",
                inputs={"X": []},
                outputs={"Out": splited_var},
                attrs={
                    "epmap": eps,
                    "trainer_id": self.trainer_id,
                    RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE
                })

        fetch_barrier_out = startup_program.global_block().create_var(
            name=framework.generate_control_dev_var_name())
        startup_program.global_block().append_op(
            type="fetch_barrier",
            inputs={},
            outputs={"Out": fetch_barrier_out},
            attrs={
                "endpoints": self.pserver_endpoints,
                "trainer_id": self.trainer_id,
                RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE
            })

        for varname, splited_var in six.iteritems(self.param_var_mapping):
            # add concat ops to merge splited parameters received from parameter servers.
            if len(splited_var) <= 1:
                continue
            # NOTE: if enable memory optimization, origin vars maybe removed.
            if varname in startup_program.global_block().vars:
                orig_param = startup_program.global_block().vars[varname]
            else:
                origin_param_var = self.origin_program.global_block().vars[
                    varname]
                orig_param = startup_program.global_block().create_var(
                    name=varname,
                    persistable=origin_param_var.persistable,
                    type=origin_param_var.type,
                    dtype=origin_param_var.dtype,
                    shape=origin_param_var.shape)
            startup_program.global_block().append_op(
                type="concat",
                inputs={"X": splited_var},
                outputs={"Out": [orig_param]},
                attrs={"axis": 0})

        dummy_output = startup_program.global_block().create_var(
            name=framework.generate_control_dev_var_name())
        param_init = startup_program.global_block().create_var(
            name="param_init")
        startup_program.global_block().append_op(
            type="send",
            inputs={"X": [param_init]},
            outputs={"Out": dummy_output},
            attrs={
                "send_varnames": [param_init.name]}
        )
        return startup_program

    def get_trainer_program(self, wait_port=True):
        # delete table init op
        if self.has_distributed_lookup_table:
            table_var = self.startup_program.global_block().vars[
                self.table_name]
            table_param_init_op = []
            for op in self.startup_program.global_block().ops:
                if self.table_name in op.output_arg_names:
                    table_param_init_op.append(op)
            init_op_num = len(table_param_init_op)
            if init_op_num != 1:
                raise ValueError("table init op num should be 1, now is " + str(
                    init_op_num))
            table_init_op = table_param_init_op[0]
            self.startup_program.global_block().append_op(
                type="fake_init",
                inputs={},
                outputs={"Out": table_var},
                attrs={"shape": table_init_op.attr('shape')})
            delete_ops(self.startup_program.global_block(), table_param_init_op)

        self.origin_program.__str__()
        if wait_port:
            wait_server_ready(self.pserver_endpoints)
        return self.origin_program

    def get_pserver_programs(self, endpoint):
        """
        Get pserver side main program and startup program for distributed training.

        Args:
            endpoint (str): current pserver endpoint.

        Returns:
            tuple: (main_program, startup_program), of type "Program"

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                #this is an example, find available endpoints in your case
                pserver_endpoints = "192.168.0.1:6174,192.168.0.2:6174"
                current_endpoint = "192.168.0.1:6174"
                trainer_id = 0
                trainers = 4
                t = fluid.DistributeTranspiler()
                t.transpile(
                    trainer_id, pservers=pserver_endpoints, trainers=trainers)
                pserver_program, pserver_startup_program = t.get_pserver_programs(current_endpoint)
        """
        pserver_prog = self.get_pserver_program(endpoint)
        pserver_startup = self.get_pserver_startup_program(
            endpoint, pserver_program=pserver_prog)
        return pserver_prog, pserver_startup

    def get_pserver_program(self, endpoint):
        sys.stderr.write(
            "get_pserver_program() is deprecated, call get_pserver_programs() to get pserver main and startup in a single call.\n"
        )
        # step1
        pserver_program = Program()
        pserver_program.random_seed = self.origin_program.random_seed
        pserver_program._copy_dist_param_info_from(self.origin_program)
        # step2
        # step2: Create vars to receive vars at parameter servers.
        recv_inputs = []
        for v in self.param_opt_ep_mapping[endpoint]["params"]:
            self._clone_var(pserver_program.global_block(), v)

        optimize_block = []
        param_to_block_id = []
        # append op to the current block
        pre_block_idx = pserver_program.num_blocks - 1
        for idx, var in enumerate(self.param_opt_ep_mapping[endpoint]["params"]):
            per_opt_block = pserver_program._create_block(pre_block_idx)
            optimize_block.append(per_opt_block)
            var_name = var.name
            pserver_block = per_opt_block.program.global_block()
            recv_var = pserver_block.vars[var_name]
            # Todo: sparse param calc delta 
            if var.type != core.VarDesc.VarType.SELECTED_ROWS:
                delta_var_name = "%s.delta" % (recv_var.name)
                delta_var = pserver_block.create_var(
                    name=delta_var_name,
                    persistable=recv_var.persistable,
                    type=recv_var.type,
                    dtype=recv_var.dtype,
                    shape=recv_var.shape)
                
                per_opt_block.append_op(
                    type="scale",
                    inputs={"X": recv_var},
                    outputs={"Out": delta_var},
                    attrs={"scale": 1.0 / float(self.trainer_num)})
                
                per_opt_block.append_op(
                    type="elementwise_add",
                    inputs={"X":delta_var,"Y":recv_var},
                    outputs={"Out":recv_var}
                )
            param_to_block_id.append(recv_var.name + ":" + str(per_opt_block.idx))

        attrs = {
            "optimize_blocks": optimize_block,
            "endpoint": endpoint,
            "Fanin": self.trainer_num,
            "sync_mode": self.sync_mode,
            "grad_to_block_id": param_to_block_id,
        }

        # step5 append the listen_and_serv op
        pserver_program.global_block().append_op(
            type="listen_and_serv",
            inputs={'X': recv_inputs},
            outputs={},
            attrs=attrs)

        pserver_program._sync_with_cpp()
        # save pserver program to generate pserver side startup relatively.
        self.pserver_program = pserver_program
        return pserver_program

    def get_pserver_startup_program(self,
                            endpoint,
                            pserver_program=None,
                            startup_program=None):
        """
        **Deprecated**

        Get startup program for current parameter server.
        Modify operator input variables if there are variables that
        were split to several blocks.

        Args:
            endpoint (str): current pserver endpoint.
            pserver_program (Program): deprecated, call get_pserver_program first.
            startup_program (Program): deprecated, should pass startup_program
                when initalizing

        Returns:
            Program: parameter server side startup program.

        Examples:
        .. code-block:: python

                pserver_endpoints = "192.168.0.1:6174,192.168.0.2:6174"
                trainer_endpoints = "192.168.0.1:6174,192.168.0.2:6174"
                current_endpoint = "192.168.0.1:6174"
                trainer_id = 0
                trainers = 4

                t = fluid.DistributeTranspiler()
                t.transpile(trainer_id, pservers=pserver_endpoints, trainers=trainers)
                pserver_program = t.get_pserver_program(current_endpoint)
                pserver_startup_program = t.get_startup_program(current_endpoint,
                                                                pserver_program)
        """
        s_prog = Program()
        orig_s_prog = self.startup_program
        s_prog.random_seed = orig_s_prog.random_seed
        params = self.param_opt_ep_mapping[endpoint]["params"]

        def _get_splited_name_and_shape(varname):
            for idx, splited_param in enumerate(params):
                pname = splited_param.name
                if same_or_split_var(pname, varname) and varname != pname:
                    return pname, splited_param.shape
            return "", []

        # 1. create vars in pserver program to startup program
        pserver_vars = pserver_program.global_block().vars
        created_var_map = collections.OrderedDict()
        for _, var in six.iteritems(pserver_vars):
            tmpvar = s_prog.global_block()._clone_variable(var)
            created_var_map[var.name] = tmpvar

        # 2. rename op outputs
        for op in orig_s_prog.global_block().ops:
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
                new_inputs = self._get_input_map_from_op(pserver_vars, op)

                if op.type in [
                    "gaussian_random", "fill_constant", "uniform_random",
                    "truncated_gaussian_random"
                ]:
                    op._set_attr("shape", list(new_outputs["Out"].shape))
                s_prog.global_block().append_op(
                    type=op.type,
                    inputs=new_inputs,
                    outputs=new_outputs,
                    attrs=op.all_attrs())

        return s_prog

    def _init_splited_vars(self):
        param_list = []
        grad_list = []
        param_grad_set = set()
        # step 1. create param_list
        for p, g in self.params_grads:
            if type(p) == Parameter and p.trainable == False:
                continue
            if p.name not in param_grad_set:
                param_list.append(p)
                param_grad_set.add(p.name)
            if g.name not in param_grad_set:
                grad_list.append(g)
                param_grad_set.add(g.name)

        param_list, grad_list = self._update_dist_lookup_table_vars(
                                param_list, grad_list, self.params_grads)

        # step 2. Slice vars into numbers of piece with block_size
        # when we slice var up into blocks, we will slice the var according to
        # pserver services' count. A pserver may have two or more listening ports.
        grad_blocks = slice_variable(grad_list,
                                     len(self.pserver_endpoints),
                                     self.config.min_block_size)
        param_blocks = slice_variable(param_list,
                                      len(self.pserver_endpoints),
                                      self.config.min_block_size)
        assert (len(grad_blocks) == len(param_blocks))

        # step 3. Create splited param from split blocks
        # origin_param_name -> [splited_param_vars]
        # Todo: update _create_vars_from_blocklist
        self.param_var_mapping = self._create_vars_from_blocklist(
            self.origin_program, param_blocks)

        for origin_name, splited_vars in self.param_var_mapping.items():
            origin_var = self.origin_program.global_block().var(origin_name)
            self.vars_info[origin_name] = collections.OrderedDict()
            self.vars_info[origin_name]["var_names"] = []
            vars_section = self._get_splited_var_sections(splited_vars)
            self.vars_info[origin_name]["sections"] = [str(i) for i in vars_section]
            self.vars_info[origin_name]["epmap"] = []
            for splited_var in splited_vars:
                is_slice, block_id, offset = self._get_slice_var_info(splited_var)
                self.vars_overview.add_distributed_var(
                    origin_var=origin_var,
                    slice_var=splited_var,
                    block_id=block_id,
                    offset=offset,
                    is_slice=is_slice,
                    vtype="Param")
                self.vars_info[origin_name]["var_names"].append(splited_var.name)
                self.split_to_origin_mapping[splited_var.name] = origin_name

        # step 4. Create mapping of endpoint -> split var to create pserver side program
        self.param_opt_ep_mapping = collections.OrderedDict()
        [
            self.param_opt_ep_mapping.update({
                ep: {
                    "params": [],
                }
            }) for ep in self.pserver_endpoints
        ]


    def _init_optimize_mapping(self):
        optimize_list = []
        optimize_to_param = dict()
        param_to_optimize = dict()

        for op in self.optimize_ops:
            if op.type == "sgd":
                origin_name = op.output("ParamOut")
                var = self.origin_program.global_block().var(origin_name[0])
                optimize_list.append(var.name)
                optimize_to_param[var.name] = var.name
                param_to_optimize[var.name] = var.name
        self._param_to_optimize = param_to_optimize
        self._optimize_to_param = optimize_to_param
        self._optimize_var_list = optimize_list