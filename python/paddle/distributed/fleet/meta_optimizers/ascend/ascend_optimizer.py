import os
import numpy as np
from collections import namedtuple

import paddle.fluid.framework as framework
from paddle.fluid.optimizer import Optimizer
import paddle.fluid.core as core
from ascend import ascend_parser

HcomGroupConfig = namedtuple('HcomGroupConfig', ['name', 'nranks', 'rank_ids'])

class AscendIRParser(object):
    def __init__(self):
        self.graph_idx = 0
        self.hcom_endpoints = {}
        self.groups_to_create = []
        
    def _construct_input_map(self, input_varlist):
        ret_map = {}
        ge_in_operator = []
        for id, var in enumerate(input_varlist):
            if var.is_data: # input data
                print("_construct_input_map for %d input var[%s]" % (id, var.name))
                ge_input = core.GEOperatorFactory.create_operator(
                    var.name, "Data").set_attr_int32("index", id)
                ret_map[var.name] = ge_input
                ge_in_operator.append(ge_input)
            else: # param
                print("_construct_input_map for %d param var[%s]" % (id, var.name))
                ge_input = core.GEOperatorFactory.create_operator(var.name, "Variable")
                ge_input.update_output_desc("y", 
                                                core.GETensorDesc(
                                                    core.GEShape(var.shape),
                                                    core.GEFormat.FORMAT_ND, 
                                                    core.GEDataType.DT_FLOAT))
                ret_map[var.name] = ge_input
        return ge_in_operator, ret_map

    def parse_op(self, op):
        if op.type == 'c_gen_nccl_id':
            endpoint = op.attr("endpoint")
            other_endpoints = op.attr("other_endpoints")
            rank = op.attr("rank")

            nccl_id = op.output_arg_names[0]

            # c_gen_nccl_id operator splits endpoints into local endpoint and other_endpoints
            # we should combine these together to produce world_rank_ids 
            self.hcom_endpoints[nccl_id] = other_endpoints[:]
            self.hcom_endpoints[nccl_id].insert(rank, endpoint)

            print("nccl_id (%s) registered endpoints %s" % (nccl_id, self.hcom_endpoints[nccl_id]))
        elif op.type == 'c_comm_init':
            nccl_id = op.input_arg_names[0]
            nranks = op.attr("nranks")
            assert nranks == len(self.hcom_endpoints[nccl_id]), "nranks doesn't match endpoint count"
            rank = op.attr("rank")
            ring_id = op.attr("ring_id")

            group_name = "hcom_group_" + str(ring_id)
            global_rank_ids = [self._endpoint_to_world_rank_id(endpoint) for endpoint in self.hcom_endpoints[nccl_id]]
            self.groups_to_create.append(HcomGroupConfig(name=group_name, nranks=nranks, rank_ids=global_rank_ids))
            print("append to create group: %s, with rank_ids: %s" % (group_name, global_rank_ids))
        elif op.type in ascend_parser.registerd_op:
            print("op[%s] has been registered" % (op.type))
            op_parser = self.parser_factory.create_parse(ascend_parser.registerd_op[op.type])
            op_parser.apply(op)
        else:
            print("op[%s] has not been registered, parse failed..." % (op.type))
            
    def _parse_program(self, graph_name, program, input_varlist=[], fetch_list=[]):
        begin_graph_idx = self.graph_idx
        ge_in_operator = []
        ge_out_operator = []
        self.var2geop = {}

        block = program.global_block()
        if len(block.ops) == 0:
            print("there is no ops in program %s" % (graph_name))
            return []

        graph = core.GEGraph(graph_name)
        print("begin parse %s" % (graph_name)) 

        ge_in_operator, self.var2geop = self._construct_input_map(input_varlist)
        
        self.parser_factory = ascend_parser.AscendParserFactory(graph, self.var2geop)
        for i, curop in list(enumerate(block.ops)):
            self.parse_op(curop)
           
        for e in fetch_list:
            name = e
            if not isinstance(e, str):
                try:
                    name = e.name
                except AttributeError:
                    print(e)
            ge_out_operator.append(self.var2geop[name])
         
        for varname, geop in self.var2geop.items():
            if varname.startswith("geinput"):
                ge_in_operator.append(geop)
        #print("ge_in_operator: ", ge_in_operator)
        #print("ge_out_operator: ", ge_out_operator)       
        graph.set_inputs(ge_in_operator).set_outputs(ge_out_operator)

        op_num = len(block.ops)
        for i in range(op_num - 1, -1, -1):
            block._remove_op(i)
        
        #if self.graph_idx == 0: # hack for startup program
        #    fetch_list = [block.var("learning_rate_0")]
        
        input_varlist = [var for var in input_varlist if var.is_data]
        
        block.append_op(
            type="ascend_trigger",
            inputs={"FeedList": input_varlist},
            outputs={"FetchList": fetch_list},
            attrs={'graph_idx': begin_graph_idx + i})
        self.graph_idx += 1
        return graph

    def parse_program(self, startup_program, main_program, input_varlist, fetch_list):
        startup_graph = self._parse_program("startup", startup_program)
        main_graph = self._parse_program("main", main_program, input_varlist, fetch_list)
        return startup_graph, main_graph


# AscendOptimizer is a wrapper for basic optimizer now
# We will make it part of fleet meta_optimizer in the future
class AscendOptimizer(Optimizer):
    def __init__(self, optimizer, fetch_list=[]):
        self.inner_opt = optimizer
        self.fetch_list = fetch_list
        
    def __del__(self):
        core.ge_finalize()

    def _can_apply(self):
        if not self.user_defined_strategy.ascend:
            return False
        # TODO(hutuxian): other check here
        return True

    def _disable_strategy(self, dist_strategy):
        dist_strategy.ascend = False
        dist_strategy.ascend_configs = {}

    def _get_input_varlist(self, program):
        ret_list = []
        for var in program.list_vars():
            if var.is_data or var.persistable:
                ret_list.append(var)
        return ret_list

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None,
                 auto_dp=False):
        minimized = None
        if self.inner_opt:
            minimized = self.inner_opt.minimize(loss, startup_program=startup_program)

        self.ascend_instance = core.AscendInstance()
        
        # Config about Graph Engine can be found in https://support.huaweicloud.com/
        config = {"ge.exec.deviceId": "0", "ge.graphRunMode": "1", "ge.exec.precision_mode": "must_keep_origin_dtype"} # "allow_mix_precision"}
        print("ge_initialize config:", config)
        core.ge_initialize(config)

        # Init Session
        self.ascend_instance.init_global_resources()  # add whatever parameters here to init
        
        main_block = loss.block
        self.parser = AscendIRParser()

        input_varlist = self._get_input_varlist(main_block.program)
        #print("input_varlist: ", input_varlist)

        startup_graph, main_graph = self.parser.parse_program(
            startup_program, main_block.program, input_varlist, self.fetch_list)
        
        self.ascend_instance.add_ascend_subgraph(0, startup_graph)
        self.ascend_instance.add_ascend_subgraph(1, main_graph)

        return minimized
