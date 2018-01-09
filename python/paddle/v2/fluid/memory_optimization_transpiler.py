from collections import defaultdict
import framework
from framework import Program, default_main_program, Parameter, Variable
import backward
from backward import _rename_arg_


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
        # print(block_size)

        # TODO(qijun) handle Program with if/while operators
        self.global_block = program_desc.block(0)
        self.op_size = self.global_block.op_size()
        # print(self.op_size)

        op_node_connections = [(i, i + 1) for i in range(self.op_size - 1)]
        self._add_connections(op_node_connections)

        # print(self._succesors)
        # print(self._presucessors)

        self.ops = [self.global_block.op(i) for i in range(self.op_size)]

        for i in range(self.op_size):
            self._uses[i].update(self.ops[i].input_arg_names())
            self._defs[i].update(self.ops[i].output_arg_names())

        # print self._uses
        # print self._defs

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

        # print(self._live_in)
        # print(self._live_out)

    def _get_diff(self, a, b):
        u = a & b
        return a - u, b - u

    def memory_optimize(self):
        self._build_graph()
        self._dataflow_analyze()
        self.pool = []
        for i in range(self.op_size):
            # print self.ops[i].type()
            if self.pool:
                out_pair = [(x, self.global_block.var(str(x)).shape())
                            for x in self._defs[i]]
                # print self.ops[i].type()
                # print out_pair
                # print self.pool
                for x, x_shape in out_pair:
                    for index, cache_pair in enumerate(self.pool):
                        cache_var = cache_pair[0]
                        cache_shape = cache_pair[1]
                        if x_shape == cache_shape:
                            print(
                                "hit cache !!!! pool index is %d, var name is %s, cache var name is %s, var shape is %s "
                                % (index, x, cache_var, str(cache_shape)))
                            self.pool.pop(index)
                            _rename_arg_(self.ops, x, cache_var, begin_idx=i)
                            self._dataflow_analyze()
                            break

            in_diff, out_diff = self._get_diff(self._live_in[i],
                                               self._live_out[i])
            can_optimize = filter(
                lambda x: not self.global_block.var(str(x)).persistable(),
                in_diff)
            if can_optimize:
                # print self.ops[i].type(), in_diff, can_optimize
                for var_name in can_optimize:
                    self.pool.append((
                        var_name, self.global_block.var(str(var_name)).shape()))
            # print("\n")

    def get_program(self):
        return self._program


def memory_optimize(input_program):
    graph = ControlFlowGraph(input_program)
    graph.memory_optimize()
    result_program = graph.get_program()
    return result_program
