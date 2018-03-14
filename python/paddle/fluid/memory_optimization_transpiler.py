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

sub_block_ops = [
    "while", "while_grad", "parallel_do", "parallel_do_grad",
    "conditional_block", "conditional_block_grad"
]

PRINT_LOG = False


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

    def _check_var_validity(self, block_desc, x, is_forward):
        if str(x) == "@EMPTY@":
            return False
        if not self._has_var(block_desc, x, is_forward):
            return False
        if self._find_var(block_desc, x, is_forward).persistable():
            return False
        if self._find_var(block_desc, x,
                          is_forward).type() != core.VarDesc.VarType.LOD_TENSOR:
            return False
        if x in self._skip_opt:
            return False
        if not self._find_var(block_desc, x, is_forward).shape():
            return False
        return True

    def _update_skip_opt_set(self):
        for i in range(self.op_size):
            op = self._ops[i]
            if op.type() == "fill_constant" and op.attr("force_cpu") == True:
                self._skip_opt.update(op.output_arg_names())

    def dataflow_analyze(self):
        self._dataflow_analyze()

    def find_variable_liveness_op(self, var_name):
        for i in range(self.op_size):
            op = self._ops[i]
            in_diff, out_diff = self._get_diff(self._live_in[i],
                                               self._live_out[i])
            if var_name in in_diff:
                return i

    def release_memory(self):
        self._dataflow_analyze()
        self._update_skip_opt_set()
        fwd_id = 0
        bwd_id = 0
        for i in range(self.op_size):
            op = self._ops[i]
            if op.type() in sub_block_ops:
                continue
            block_desc = op.block()
            is_forward = i < self._forward_num
            in_diff, out_diff = self._get_diff(self._live_in[i],
                                               self._live_out[i])
            can_optimize = filter(
                lambda x: self._check_var_validity(block_desc, x, is_forward),
                in_diff)
            if can_optimize:
                index = i + fwd_id + 1 if is_forward else i - self._forward_num + bwd_id + 1
                delete_op = block_desc.insert_op(index)
                delete_op.set_type("delete_var")
                delete_op.set_input("X", can_optimize)
                if is_forward:
                    fwd_id += 1
                else:
                    bwd_id += 1

    def memory_optimize(self, level=0):
        def compare_shape(x_shape, cache_shape, opt_level):
            if opt_level == 0:
                return x_shape == cache_shape
            if opt_level == 1:
                if (x_shape[0] == -1) ^ (cache_shape[0] == -1):
                    return False
                x_size = abs(reduce(lambda x, y: x * y, x_shape))
                cache_size = abs(reduce(lambda x, y: x * y, cache_shape))
                if x_size <= cache_size:
                    return True
            return False

        self._dataflow_analyze()
        self._update_skip_opt_set()
        self.pool = []
        for i in range(self.op_size):
            op = self._ops[i]
            if op.type() in sub_block_ops:
                continue
            block_desc = op.block()
            self.current_block_desc = block_desc
            is_forward = i < self._forward_num
            if self.pool:
                defs_can_optimize = filter(
                    lambda x: self._check_var_validity(block_desc, x, is_forward),
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
                        if compare_shape(x_shape, cache_shape, level):
                            if self._has_var(block_desc, cache_var, is_forward):
                                x_dtype = self._find_var(block_desc, x,
                                                         is_forward).dtype()
                                cache_dtype = self._find_var(
                                    block_desc, cache_var, is_forward).dtype()
                                # TODO(qijun): actually, we should compare dtype_to_size[x_dtype]
                                # and dtype_to_size[cache_dtype]
                                if x_dtype == cache_dtype:
                                    if PRINT_LOG:
                                        print(
                                            ("Hit Cache !!!! cache pool index "
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
                lambda x: self._check_var_validity(block_desc, x, is_forward),
                in_diff)
            if can_optimize:
                for var_name in can_optimize:
                    self.pool.append((var_name, self._find_var(
                        block_desc, var_name, is_forward).shape()))


def _process_sub_block_pair(pdesc, sub_block_pair):
    ops_list = []
    block_desc = pdesc.block(0)
    op_size = block_desc.op_size()
    for fwd_op, bwd_op in sub_block_pair:
        sub_block_ids = []
        grad_sub_block_ids = []
        sub_block_id_pair = []
        sub_op_dict = {}
        for i in range(op_size):
            op = block_desc.op(i)
            if op.type() == fwd_op:
                sub_block_ids.append(op.attr("sub_block").id)
                sub_op_dict[op.attr("sub_block").id] = op
            elif op.type() == bwd_op:
                grad_sub_block_ids.append(op.attr("sub_block").id)
                sub_op_dict[op.attr("sub_block").id] = op

        # Find fwd_op/bwd_op block pair
        for grad_id in grad_sub_block_ids:
            fwd_id = pdesc.block(grad_id).get_forward_block_idx()
            if fwd_id in sub_block_ids:
                sub_block_id_pair.append((fwd_id, grad_id))
                sub_block_ids.remove(fwd_id)

        # Get fwd_op/bwd_op block ops
        for fwd_id, grad_id in sub_block_id_pair:
            sub_block_ops = []
            sub_block = pdesc.block(fwd_id)
            block_op_size = sub_block.op_size()
            for i in range(block_op_size):
                sub_block_ops.append(sub_block.op(i))

            grad_sub_block = pdesc.block(grad_id)
            grad_sub_block_op_size = grad_sub_block.op_size()
            for i in range(grad_sub_block_op_size):
                sub_block_ops.append(grad_sub_block.op(i))

            sub_op_output = set()
            sub_op_output.update(sub_op_dict[fwd_id].output_arg_names())
            sub_op_output.update(sub_op_dict[grad_id].output_arg_names())
            ops_list.append((sub_block_ops, block_op_size, sub_op_output))

        # Process rest fwd_op block ops
        for fwd_id in sub_block_ids:
            sub_block_ops = []
            sub_block = pdesc.block(fwd_id)
            sub_block_op_size = sub_block.op_size()
            for i in range(sub_block_op_size):
                sub_block_ops.append(sub_block.op(i))
            sub_op_output = set()
            sub_op_output.update(sub_op_dict[fwd_id].output_arg_names())
            ops_list.append((sub_block_ops, sub_block_op_size, sub_op_output))
    return ops_list


def _get_cfgs(input_program):
    ops_list = []
    pdesc = input_program.get_desc()
    block_desc = pdesc.block(0)
    op_size = block_desc.op_size()
    # Get global block ops
    ops_list.append(
        ([block_desc.op(i) for i in range(op_size)], op_size, set()))

    sub_block_pair = [("while", "while_grad"), ("parallel_do",
                                                "parallel_do_grad"),
                      ("conditional_block", "conditional_block_grad")]

    ops_list.extend(_process_sub_block_pair(pdesc, sub_block_pair))

    cfgs = [
        ControlFlowGraph(input_program, ops, forward_num, skip_opt)
        for ops, forward_num, skip_opt in ops_list
    ]
    return cfgs


def memory_optimize(input_program, print_log=False, level=0):
    global PRINT_LOG
    PRINT_LOG = print_log
    cfgs = _get_cfgs(input_program)
    for cfg in cfgs:
        cfg.memory_optimize(level)


def release_memory(input_program):
    cfgs = _get_cfgs(input_program)
    for cfg in cfgs:
        cfg.release_memory()


activation_ops = [
    "sigmoid", "logsigmoid", "exp", "relu", "tanh", "tanh_shrink", "softshrink",
    "sqrt", "abs", "ceil", "floor", "round", "reciprocal", "log", "square",
    "softplus", "softsign", "brelu", "leaky_relu", "soft_relu", "elu", "relu6",
    "pow", "stanh", "hard_shrink", "thresholded_relu", "hard_sigmoid", "swish"
]


def _check_match_pattern(ops, pattern):
    match_flag = True
    for j in range(len(ops)):
        if pattern[j] == "activation":
            if ops[j].type() not in activation_ops:
                match_flag = False
                break
        else:
            if ops[j].type() != pattern[j]:
                match_flag = False
                break
    return match_flag


def _find_forward_index(block_desc):
    pattern = ["mean", "fill_constant", "mean_grad"]
    op_size = block_desc.op_size()
    for i in range(op_size - len(pattern) + 1):
        ops = [block_desc.op(i + j) for j in range(len(pattern))]
        match_flag = _check_match_pattern(ops, pattern)
        if match_flag:
            return i


def recomputation(input_program, pattern="activation"):
    pdesc = input_program.get_desc()
    block_desc = pdesc.block(0)
    op_size = block_desc.op_size()

    match_ops = []
    for i in range(op_size):
        op = block_desc.op(i)
        if pattern == "activation":
            if op.type() in activation_ops:
                match_ops.append(op)

    forward_index = _find_forward_index(block_desc)
    forward_ops = [block_desc.op(i) for i in range(forward_index)]
    forward_cfg = ControlFlowGraph(input_program, forward_ops, forward_index,
                                   set())
    forward_cfg.dataflow_analyze()

    match_indexs = []
    for op in match_ops:
        # print op.type()
        # print op.output("Out")[0]
        index = forward_cfg.find_variable_liveness_op(op.output("Out")[0])
        # print index
        match_indexs.append(index)

    forward_hit = zip(match_ops, match_indexs)
    forward_hit = sorted(forward_hit, key=lambda x: x[1])

    for i, (fwd_op, fwd_index) in enumerate(forward_hit):
        delete_op = block_desc.insert_op(fwd_index + i + 1)
        delete_op.set_type("delete_var")
        delete_op.set_input("X", fwd_op.output("Out"))

    current_op_size = block_desc.op_size()
    for i, (fwd_op, fwd_index) in enumerate(forward_hit):
        bwd_index = 0
        for j in range(forward_index + len(forward_hit), current_op_size):
            bwd_op = block_desc.op(j)
            if fwd_op.output("Out")[0] in bwd_op.input_arg_names():
                new_op = block_desc.insert_op(j + bwd_index)
                new_op.copy_from(fwd_op)
                bwd_index += 1
                break
