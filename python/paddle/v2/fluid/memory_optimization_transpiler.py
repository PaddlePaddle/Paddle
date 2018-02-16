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

from collections import defaultdict
import framework
from framework import Program, default_main_program, Parameter, Variable
import backward
from backward import _rename_arg_
from . import core

dtype_to_size = {
    core.VarDesc.VarType.FP16: 2,
    core.VarDesc.VarType.FP32: 4,
    core.VarDesc.VarType.FP64: 8,
    core.VarDesc.VarType.INT16: 2,
    core.VarDesc.VarType.INT32: 4,
    core.VarDesc.VarType.INT64: 8,
    core.VarDesc.VarType.BOOL: 1
}


class ControlFlowGraph(object):
    def __init__(self, Program, ops, forward_num, skip_opt):
        self._program = Program
        self._ops = ops
        self._forward_num = forward_num
        self._successors = defaultdict(set)
        self._presuccessors = defaultdict(set)
        self._uses = defaultdict(set)
        self._defs = defaultdict(set)
        self._live_in = defaultdict(set)
        self._live_out = defaultdict(set)
        self._skip_opt = skip_opt

    def _add_connections(self, connections):
        for node1, node2 in connections:
            self._add(node1, node2)

    def _add(self, node1, node2):
        self._successors[node1].add(node2)
        self._presuccessors[node2].add(node1)

    def _build_graph(self):
        self.op_size = len(self._ops)
        op_node_connections = [(i, i + 1) for i in range(self.op_size - 1)]
        self._add_connections(op_node_connections)
        for i in range(self.op_size):
            self._uses[i].update(self._ops[i].input_arg_names())
            self._defs[i].update(self._ops[i].output_arg_names())

    def _update_graph(self, old_name, new_name, begin_idx=0):
        for i in range(begin_idx, self.op_size):
            if old_name in self._uses[i]:
                self._uses[i].remove(old_name)
                self._uses[i].add(new_name)
            if old_name in self._defs[i]:
                self._defs[i].remove(old_name)
                self._defs[i].add(new_name)
            if old_name in self._live_in[i]:
                self._live_in[i].remove(old_name)
                self._live_out[i].add(new_name)
            if old_name in self._live_out[i]:
                self._live_out[i].remove(old_name)
                self._live_out[i].add(new_name)

    def _reach_fixed_point(self, live_in, live_out):
        if len(live_in) != len(self._live_in):
            return False
        if len(live_out) != len(self._live_out):
            return False
        for i in range(self.op_size):
            if live_in[i] != self._live_in[i]:
                return False
        for i in range(self.op_size):
            if live_out[i] != self._live_out[i]:
                return False
        return True

    def _dataflow_analyze(self):
        self._build_graph()
        live_in = defaultdict(set)
        live_out = defaultdict(set)
        while True:
            for i in range(self.op_size, 0, -1):
                live_in[i] = set(self._live_in[i])
                live_out[i] = set(self._live_out[i])
                for s in self._successors[i]:
                    self._live_out[i] |= self._live_in[s]
                self._live_in[i] = self._uses[i] | (
                    self._live_out[i] - self._defs[i])
            if self._reach_fixed_point(live_in, live_out):
                break

    def _get_diff(self, a, b):
        u = a & b
        return a - u, b - u

    def _has_var(self, block_desc, var_name, is_forward):
        if is_forward:
            return block_desc.has_var(str(var_name))
        else:
            return block_desc.has_var_recursive(str(var_name))

    def _find_var(self, block_desc, var_name, is_forward):
        if is_forward:
            return block_desc.find_var(str(var_name))
        else:
            return block_desc.find_var_recursive(str(var_name))

    def memory_optimize(self):
        def check_var_validity(block_desc, x, is_forward):
            if str(x) == "@EMPTY@":
                return False
            if not self._has_var(block_desc, x, is_forward):
                return False
            if self._find_var(block_desc, x, is_forward).persistable():
                return False
            if self._find_var(
                    block_desc, x,
                    is_forward).type() != core.VarDesc.VarType.LOD_TENSOR:
                return False
            if x in self._skip_opt:
                return False
            if not self._find_var(block_desc, x, is_forward).shape():
                return False
            return True

        self._build_graph()
        self._dataflow_analyze()
        self.pool = []
        for i in range(self.op_size):
            op = self._ops[i]
            if op.type() == "while" or op.type() == "while_grad":
                continue
            block_desc = op.block()
            is_forward = i < self._forward_num
            if self.pool:
                defs_can_optimize = filter(
                    lambda x: check_var_validity(block_desc, x, is_forward),
                    self._defs[i])
                out_pair = [
                    (x, self._find_var(block_desc, x, is_forward).shape())
                    for x in defs_can_optimize
                ]
                for x, x_shape in out_pair:
                    # If x is both in uses and defs, it can not be optimized!
                    if x in self._uses[i]:
                        continue
                    for index, cache_pair in enumerate(self.pool):
                        cache_var = cache_pair[0]
                        cache_shape = cache_pair[1]
                        if x_shape == cache_shape:
                            if self._has_var(block_desc, cache_var, is_forward):
                                x_dtype = self._find_var(block_desc, x,
                                                         is_forward).dtype()
                                cache_dtype = self._find_var(
                                    block_desc, cache_var, is_forward).dtype()
                                # TODO(qijun): actually, we should compare dtype_to_size[x_dtype]
                                # and dtype_to_size[cache_dtype]
                                if x_dtype == cache_dtype:
                                    print(("Hit Cache !!!! cache pool index "
                                           "is %d, var name is %s, "
                                           "cached var name is %s, "
                                           "var shape is %s ") %
                                          (index, x, cache_var,
                                           str(cache_shape)))
                                    self.pool.pop(index)
                                    if x == cache_var:
                                        break
                                    _rename_arg_(
                                        self._ops, x, cache_var, begin_idx=i)
                                    self._program.block(block_desc.id).var(
                                        str(x)).desc = self._find_var(
                                            block_desc, cache_var, is_forward)
                                    self._update_graph(
                                        x, cache_var, begin_idx=i)
                                    break

            in_diff, out_diff = self._get_diff(self._live_in[i],
                                               self._live_out[i])
            can_optimize = filter(
                lambda x: check_var_validity(block_desc, x, is_forward),
                in_diff)
            if can_optimize:
                for var_name in can_optimize:
                    self.pool.append((var_name, self._find_var(
                        block_desc, var_name, is_forward).shape()))


def get_cfgs(input_program):
    ops_list = []
    pdesc = input_program.get_desc()
    block_desc = pdesc.block(0)
    op_size = block_desc.op_size()
    # Get global block ops
    ops_list.append(
        ([block_desc.op(i) for i in range(op_size)], op_size, set()))

    while_sub_block_ids = []
    while_grad_sub_block_ids = []
    while_block_id_pair = []
    while_op_dict = {}

    for i in range(op_size):
        op = block_desc.op(i)
        if op.type() == "while":
            while_sub_block_ids.append(op.attr("sub_block").id)
            while_op_dict[op.attr("sub_block").id] = op
        elif op.type() == "while_grad":
            while_grad_sub_block_ids.append(op.attr("sub_block").id)
            while_op_dict[op.attr("sub_block").id] = op

    # Find while/while_grad block pair
    for grad_id in while_grad_sub_block_ids:
        parent_id = pdesc.block(grad_id).parent
        if parent_id in while_sub_block_ids:
            while_block_id_pair.append((parent_id, grad_id))
            while_sub_block_ids.remove(parent_id)

    # Get while/while_grad block ops
    for parent_id, grad_id in while_block_id_pair:
        while_block_ops = []
        while_block = pdesc.block(parent_id)
        while_block_op_size = while_block.op_size()
        for i in range(while_block_op_size):
            while_block_ops.append(while_block.op(i))

        while_grad_block = pdesc.block(grad_id)
        while_grad_block_op_size = while_grad_block.op_size()
        for i in range(while_grad_block_op_size):
            while_block_ops.append(while_grad_block.op(i))

        while_op_output = set()
        while_op_output.update(while_op_dict[parent_id].output_arg_names())
        while_op_output.update(while_op_dict[grad_id].output_arg_names())

        ops_list.append((while_block_ops, while_block_op_size, while_op_output))

    # Process rest while block ops
    for parent_id in while_sub_block_ids:
        while_block_ops = []
        while_block = pdesc.block(parent_id)
        while_block_op_size = while_block.op_size()
        for i in range(while_block_op_size):
            while_block_ops.append(while_block.op(i))

        while_op_output = set()
        while_op_output.update(while_op_dict[parent_id].output_arg_names())

        ops_list.append((while_block_ops, while_block_op_size, while_op_output))

    cfgs = [
        ControlFlowGraph(input_program, ops, forward_num, skip_opt)
        for ops, forward_num, skip_opt in ops_list
    ]
    return cfgs


def memory_optimize(input_program):
    cfgs = get_cfgs(input_program)
    for cfg in cfgs:
        cfg.memory_optimize()
