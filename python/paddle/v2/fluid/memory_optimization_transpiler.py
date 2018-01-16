from collections import defaultdict
import framework
from framework import Program, default_main_program, Parameter, Variable
import backward
from backward import _rename_arg_
from . import core

dtype_to_size = {
    core.DataType.FP16: 2,
    core.DataType.FP32: 4,
    core.DataType.FP64: 8,
    core.DataType.INT16: 2,
    core.DataType.INT32: 4,
    core.DataType.INT64: 8,
    core.DataType.BOOL: 1
}

kCondition = "Condition"
kStepScopes = "StepScopes"


class ControlFlowGraph(object):
    def __init__(self, Program):
        self._program = Program
        self._succesors = defaultdict(set)
        self._presucessors = defaultdict(set)
        self._uses = defaultdict(set)
        self._defs = defaultdict(set)
        self._live_in = defaultdict(set)
        self._live_out = defaultdict(set)
        self._ops = []

    def _add_connection(self, node1, node2):
        self._succesors[node1].add(node2)
        self._presucessors[node2].add(node1)

    def _build_graph_imp(self, block_idx, op_idx):
        program_desc = self._program.get_desc()
        block_desc = program_desc.block(block_idx)
        op_size = block_desc.op_size()
        for i in range(op_size):
            op = block_desc.op(i)
            if i != op_size:
                self._add_connection(op_idx, op_idx + 1)
            self._ops.append(op)
            if op.type() != "while" and op.type() != "while_grad":
                self._uses[op_idx].update(op.input_arg_names())
                self._defs[op_idx].update(op.output_arg_names())
                op_idx += 1
            else:
                if op.type() == "while":
                    self._uses[op_idx].update(op.input(kCondition))
                    self._defs[op_idx].update(op.output(kStepScopes))
                elif op.type() == "while_grad":
                    self._uses[op_idx].update(op.input(kStepScopes))
                sub_block_id = op.attr("sub_block")
                op_idx += 1
                new_op_idx = self._build_graph_imp(sub_block_id, op_idx)
                self._add_connection(new_op_idx, op_idx - 1)
                self._add_connection(op_idx - 1, new_op_idx + 1)
        return op_idx

    def _build_graph(self):
        self._build_graph_imp(0, 0)
        self.op_size = len(self._ops)
        print "build graph done ", self.op_size

    def clear_graph(self):
        self._succesors.clear()
        self._presucessors.clear()
        self._uses.clear()
        self._defs.clear()
        self._live_in.clear()
        self._live_out.clear()
        del self.pool[:]

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
        live_in = defaultdict(set)
        live_out = defaultdict(set)
        while True:
            for i in range(self.op_size):
                live_in[i] = set(self._live_in[i])
                live_out[i] = set(self._live_out[i])
                self._live_in[i] = self._uses[i] | (
                    self._live_out[i] - self._defs[i])
                for s in self._succesors[i]:
                    self._live_out[i] |= self._live_in[s]

            if self._reach_fixed_point(live_in, live_out):
                break

    def _get_diff(self, a, b):
        u = a & b
        return a - u, b - u

    def memory_optimize(self):
        self._build_graph()
        self._dataflow_analyze()
        self.pool = []
        for i in range(self.op_size):
            op = self._ops[i]
            block_desc = op.block()
            if self.pool:
                defs_can_optimize = filter(
                    lambda x: str(x) != "@EMPTY@" and
                    block_desc.has_var_recursive(str(x)) and
                    block_desc.find_var_recursive(str(x)).type() == core.VarDesc.VarType.LOD_TENSOR,
                    self._defs[i])
                out_pair = [(x, block_desc.find_var_recursive(str(x)).shape())
                            for x in defs_can_optimize]
                for x, x_shape in out_pair:
                    if not block_desc.find_var_recursive(str(x)).persistable():
                        for index, cache_pair in enumerate(self.pool):
                            cache_var = cache_pair[0]
                            cache_shape = cache_pair[1]
                            if x_shape == cache_shape:
                                x_dtype = block_desc.find_var_recursive(
                                    str(x)).dtype()
                                cache_dtype = block_desc.find_var_recursive(
                                    str(cache_var)).dtype()
                                # TODO(qijun): actually, we should compare dtype_to_size[x_dtype]
                                # and dtype_to_size[cache_dtype]
                                if x_dtype == cache_dtype:
                                    print(
                                        ("Hit Cache !!!! cache pool index "
                                         "is %d, var name is %s, "
                                         "cached var name is %s, "
                                         "var shape is %s ") %
                                        (index, x, cache_var, str(cache_shape)))
                                    self.pool.pop(index)
                                    _rename_arg_(
                                        self._ops, x, cache_var, begin_idx=i)
                                    block_id = block_desc.id()
                                    self._program.block(block_id).var(str(
                                        x)).desc = block_desc.find_var_recursive(
                                            str(cache_var))
                                    self._update_graph(
                                        x, cache_var, begin_idx=i)
                                    break

            in_diff, out_diff = self._get_diff(self._live_in[i],
                                               self._live_out[i])
            can_optimize = filter(
                lambda x: str(x) != "@EMPTY@" and
                block_desc.has_var_recursive(str(x)) and
                not block_desc.find_var_recursive(str(x)).persistable(),
                in_diff)
            can_optimize = filter(
                lambda x: block_desc.find_var_recursive(str(x)).type() == core.VarDesc.VarType.LOD_TENSOR,
                can_optimize)
            if can_optimize:
                for var_name in can_optimize:
                    self.pool.append(
                        (var_name,
                         block_desc.find_var_recursive(str(var_name)).shape()))

    def get_program(self):
        return self._program


def memory_optimize(input_program):
    graph = ControlFlowGraph(input_program)
    graph.memory_optimize()
    result_program = graph.get_program()
    return result_program


# def memory_optimize(input_program):
#     graph = ControlFlowGraph(input_program)
#     block_num = input_program.get_desc().num_blocks()
#     for i in range(block_num):
#         graph.memory_optimize(i)
#         graph.clear_graph()
#     result_program = graph.get_program()
#     return result_program
