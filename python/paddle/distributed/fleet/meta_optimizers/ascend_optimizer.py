# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.fluid import program_guard, layers, default_main_program
from paddle.fluid.optimizer import Momentum, SGD
from .meta_optimizer_base import MetaOptimizerBase
from .common import OpRole, OP_ROLE_KEY, CollectiveHelper, is_update_op
import paddle.fluid.core as core

import ge_stub as ge
import op_stub as op
import numpy as np


class AscendIRParser(object):
    def __init__(self):
        self.parsed_startup = None
        self.parsed_main = None

    def parse_program(self, startup_program, main_program):
        startup_subgraphs_with_id = []
        main_subgraphs_with_id = []
        # parse main program here and generate subgraph

        # A fake implementation here
        sub_graph = []
        block = startup_program.global_block()
        for i, op in list(enumerate(block.ops)):
            sub_graph.append(op.type)
        op_num = len(block.ops)
        for i in range(op_num - 1, -1, -1):
            block._remove_op(i)
        tmp_var = block.create_var(
            name="tmp", shape=[1], persistable=True, stop_gradient=True)
        block._insert_op(
            0,
            type="ascend_trigger",
            inputs={"FeedList": [tmp_var]},
            outputs={"FetchList": [tmp_var]},
            attrs={'graph_idx': 0})
        startup_subgraphs_with_id.append(sub_graph)
        sub_graph = []
        block = main_program.global_block()
        for i, op in list(enumerate(block.ops)):
            sub_graph.append(op.type)
        op_num = len(block.ops)
        for i in range(op_num - 1, -1, -1):
            block._remove_op(i)
        tmp_var = block.create_var(
            name="tmp", shape=[1], persistable=True, stop_gradient=True)
        block._insert_op(
            0,
            type="ascend_trigger",
            inputs={"FeedList": [tmp_var]},
            outputs={"FetchList": [tmp_var]},
            attrs={'graph_idx': 1})
        main_subgraphs_with_id.append(sub_graph)
        return startup_subgraphs_with_id, main_subgraphs_with_id


def gen_init_graph(graph_name, tensor_desc_list, var_name, var_values):
    def gen_tensor(tensor_shape, value):
        size = 1
        for i in range(len(tensor_shape)):
            size *= tensor_shape[i]
        np_data = np.zeros(size, dtype=np.float16)
        for i in range(size):
            np_data[i] = value

        input_tensor_desc = ge.TensorDesc(
            ge.Shape(tensor_shape), ge.FORMAT_ND, ge.DT_FLOAT16)
        tensor = ge.Tensor()
        tensor.set_tensor_desc(input_tensor_desc)
        tensor.set_data(np_data)
        return tensor

    graph = ge.Graph(graph_name)
    in_operator = []
    out_operator = []
    for i in range(len(tensor_desc_list)):
        tensor_desc_list[i].set_real_dim_cnt(tensor_desc_list[i].get_shape()
                                             .get_dim_num())
        tensor = gen_tensor(tensor_desc_list[i].get_shape().get_dims(),
                            var_values[i])

        var_const = op.Constant().set_attr_value(tensor)
        var_const.update_output_desc_y(tensor_desc_list[i])

        var_init = op.Variable(var_name[i])
        var_init.update_output_desc_y(tensor_desc_list[i])
        var_assign = op.Assign().set_input_ref(var_init).set_input_value(
            var_const)
        in_operator.append(var_init)

    graph.set_inputs(in_operator).set_outputs(out_operator)
    return graph


#generate add graph
def gen_add_graph(graph_name, var_desc_list, var_name_list):
    graph = ge.Graph(graph_name)

    data_x1_shape = op.Data("x1").set_attr_index(0)
    data_x2_shape = op.Data("x2").set_attr_index(1)

    var_x1 = op.Variable(var_name_list[0])
    var_x2 = op.Variable(var_name_list[1])
    var_x1.update_output_desc_y(var_desc_list[0])
    var_x2.update_output_desc_y(var_desc_list[1])

    graph.add_op(var_x1)
    graph.add_op(var_x2)

    add = op.Add().set_input_x1(data_x1_shape).set_input_x2(data_x2_shape)

    in_operator = [data_x1_shape, data_x2_shape]
    out_operator = [add]
    graph.set_inputs(in_operator).set_outputs(out_operator)
    graph.add_op(add)

    return graph


class MnistParser(object):
    def __init__(self):
        self.graph_idx = 0
        pass

    def _parse_program(self, program):
        begin_graph_idx = self.graph_idx
        subgraphs = []
        block = program.global_block()
        if len(block.ops) == 0:
            print("there is no ops in program")
            return []

        for i, curop in list(enumerate(block.ops)):
            if curop.type == "elementwise_add":
                var1 = block.var(curop.input_arg_names[0])
                var2 = block.var(curop.input_arg_names[1])
                shape = var1.shape
                print("shape:", shape)
                desc = ge.TensorDesc(
                    ge.Shape(shape), ge.FORMAT_ND, ge.DT_FLOAT16)
                var_tensor_desc = [desc, desc]

                # 1.1 init graph
                init_graph_id = 0
                var_name = ['x1', 'x2']
                init_var_graph = gen_init_graph("InitVarGraph", var_tensor_desc,
                                                var_name, [1, 2])
                print("init_var_graph: ", init_var_graph)
                print("Generate init graph success.")
                subgraphs.append(str(init_var_graph))
                # 1.2 add graph
                add_graph_id = 1
                add_graph = gen_add_graph("AddGraph", var_tensor_desc, var_name)
                print("Generate add graph success.")
                subgraphs.append(str(add_graph))
            elif curop.type == "mul":
                pass
            else:
                pass
        op_num = len(block.ops)
        for i in range(op_num - 1, -1, -1):
            block._remove_op(i)

        tmp_var = block.create_var(
            name="tmp", shape=[1], persistable=True, stop_gradient=True)
        for i in range(len(subgraphs)):
            block.append_op(
                type="ascend_trigger",
                inputs={"FeedList": [tmp_var]},
                outputs={"FetchList": [tmp_var]},
                attrs={'graph_idx': begin_graph_idx + i})
        return subgraphs

    def parse_program(self, startup_program, main_program):
        startup_subgraphs_with_id = self._parse_program(startup_program)
        main_subgraphs_with_id = self._parse_program(main_program)
        return startup_subgraphs_with_id, main_subgraphs_with_id


class AscendOptimizer(MetaOptimizerBase):
    def __init__(self, optimizer):
        super(AscendOptimizer, self).__init__(optimizer)
        self.inner_opt = optimizer
        self.meta_optimizers_white_list = []

    def _can_apply(self):
        if not self.user_defined_strategy.ascend:
            return False

        # TODO(hutuxian): other check here
        return True

    def _disable_strategy(self, dist_strategy):
        dist_strategy.ascend = False
        dist_strategy.ascend_configs = {}

    def minimize_impl(self,
                      loss,
                      startup_program=None,
                      parameter_list=None,
                      no_grad_set=None):
        minimized = self.inner_opt.minimize(
            loss, startup_program=startup_program)

        self.ascend_instance = core.AscendInstance()
        self.ascend_instance.init_global_resources(
        )  # add whatever parameters here to init

        main_block = loss.block
        # self.parser = AscendIRParser()
        self.parser = MnistParser()
        startup_subgraphs_with_id, main_subgraphs_with_id = self.parser.parse_program(
            startup_program, main_block.program)
        idx = 0
        for graph_with_id in startup_subgraphs_with_id:
            self.ascend_instance.add_ascend_subgraph(idx, graph_with_id)
            idx += 1
        for graph_with_id in main_subgraphs_with_id:
            self.ascend_instance.add_ascend_subgraph(idx, graph_with_id)
            idx += 1
        return minimized
