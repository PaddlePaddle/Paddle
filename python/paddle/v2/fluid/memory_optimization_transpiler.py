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


class ControlFlowGraph(object):
    def __init__(self, Program):
        self._program = Program
        self._succesors = defaultdict(set)
        self._presucessors = defaultdict(set)
        self._uses = defaultdict(set)
        self._defs = defaultdict(set)
        self._live_in = defaultdict(set)
        self._live_out = defaultdict(set)

    def _add_connections(self, connections):
        for node1, node2 in connections:
            self._add(node1, node2)

    def _add(self, node1, node2):
        self._succesors[node1].add(node2)
        self._presucessors[node2].add(node1)

    def _build_graph(self):
        program_desc = self._program.get_desc()
        block_size = program_desc.num_blocks()

        # TODO(qijun) handle Program with if/while operators
        self.global_block_desc = program_desc.block(0)
        self.op_size = self.global_block_desc.op_size()

        op_node_connections = [(i, i + 1) for i in range(self.op_size - 1)]
        self._add_connections(op_node_connections)

        self.ops = [self.global_block_desc.op(i) for i in range(self.op_size)]

        for i in range(self.op_size):
            self._uses[i].update(self.ops[i].input_arg_names())
            self._defs[i].update(self.ops[i].output_arg_names())

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
            if self.pool:
                out_pair = [(x, self.global_block_desc.var(str(x)).shape())
                            for x in self._defs[i]]
                for x, x_shape in out_pair:
                    if not self.global_block_desc.var(str(x)).persistable():
                        for index, cache_pair in enumerate(self.pool):
                            cache_var = cache_pair[0]
                            cache_shape = cache_pair[1]
                            if x_shape == cache_shape:
                                x_dtype = self.global_block_desc.var(str(
                                    x)).dtype()
                                cache_dtype = self.global_block_desc.var(
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
                                        self.ops, x, cache_var, begin_idx=i)
                                    self._program.current_block().var(str(
                                        x)).desc = self.global_block_desc.var(
                                            str(cache_var))
                                    self._update_graph(
                                        x, cache_var, begin_idx=i)
                                    break

            in_diff, out_diff = self._get_diff(self._live_in[i],
                                               self._live_out[i])
            can_optimize = filter(
                lambda x: not self.global_block_desc.var(str(x)).persistable(),
                in_diff)
            if can_optimize:
                for var_name in can_optimize:
                    self.pool.append(
                        (var_name,
                         self.global_block_desc.var(str(var_name)).shape()))

    def get_program(self):
        return self._program


def memory_optimize(input_program):
    graph = ControlFlowGraph(input_program)
    graph.memory_optimize()
    result_program = graph.get_program()
    return result_program
