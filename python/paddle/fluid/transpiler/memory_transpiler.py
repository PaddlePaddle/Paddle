# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from collections import defaultdict, OrderedDict
import functools
import paddle.fluid.core as core

import paddle.fluid.layers as layers
import paddle.fluid.optimizer as optimizer
from paddle.fluid.framework import Program, program_guard, Variable, Operator
from memory_optimization_transpiler import memory_optimize, release_memory
import paddle.fluid.layers as layers
import paddle.fluid.core as core
from paddle.fluid.framework import default_startup_program, default_main_program
from paddle.fluid.executor import Executor
from paddle.fluid.backward import append_backward
import numpy

class VarHandle(object):
    def __init__(self, name, version=0):
        self.name = name
        self.version = version
        self.place = None
        self.var = None
        self.op = None # the op generated var
        self.size = None

    def __str__(self):
        return "%s:%d" %(self.name, self.version)

    def set_handle(self, var, op):
        if not isinstance(var, Variable):
            raise ValueError("VarHandle expects Variable as arguments. But get %s", type(var))
        if not isinstance(op, Operator):
            raise ValueError("VarHandle expects Operator as arguments. But get %s", type(op))
        self.var = var
        self.op = op



class OpHandle(object):
    def __init__(self, op):
        self.op = op
        self.inputs = []
        self.outputs = []


class SSAGraph(object):
    def __init__(self, main_program=None, startup_program=None):
        self._main_program = main_program
        self._startup_program = startup_program
        self._vars = defaultdict(list) # list of generated op
        self._ops = []
        self._defs = defaultdict(set)
        self._uses = defaultdict(set)
        self._ssa_vars = defaultdict(list)
        self._inputs = defaultdict(set)
        self._outputs = defaultdict(set)
        self._doms = {} # the dominators in ssa graph
        self._preds = defaultdict(list)
        self._succs = defaultdict(list)

    def _no_output_vars(self, ops):
        inputs = set()
        outputs = set()
        for op in ops:
            for input_arg in op.input_arg_names:
                inputs.add(input_arg)
            for output_arg in op.output_arg_names:
                outputs.add(output_arg)
        return inputs - outputs

    def _find_dominators(self, post=False):
        """
        # See theoretical description in
        # http://en.wikipedia.org/wiki/Dominator_%28graph_theory%29
        # The algorithm implemented here uses a todo-list as described
        # in http://pages.cs.wisc.edu/~fischer/cs701.f08/finding.loops.html
        """
        if post:
            entries = set(self._ops[-1])
            preds_table = self._succs
            succs_table = self._preds
        else:
            entries = set([self._ops[0]])
            preds_table = self._preds
            succs_table = self._succs

        if not entries:
            raise RuntimeError("no entry points: dominator algorithm "
                               "cannot be seeded")

        doms = {}
        for e in entries:
            doms[e] = set([e])

        todo = []
        for n in self._nodes:
            if n not in entries:
                doms[n] = set(self._nodes)
                todo.append(n)

        while todo:
            n = todo.pop()
            if n in entries:
                continue
            new_doms = set([n])
            preds = preds_table[n]
            if preds:
                new_doms |= functools.reduce(set.intersection,
                                             [doms[p] for p in preds])
            if new_doms != doms[n]:
                assert len(new_doms) < len(doms[n])
                doms[n] = new_doms
                todo.extend(succs_table[n])
        self._doms = doms
        return doms

    def _to_ssa_graph(self, ops):
        def can_run(op, var_set):
            if len(op.output_arg_names) == 0:
                return True
            for input_arg in op.input_arg_names:
                if input_arg not in var_set:
                    return False
            return True
        ready_ops = list()
        ready_vars = self._no_output_vars(ops)
        for var in ready_vars:
            self._ssa_vars[var].append(VarHandle(var, 0))
        op_queue = list(ops)
        while op_queue:
            op = op_queue.pop(0)
            if can_run(op, ready_vars):
                ready_ops.append(op)
                for output_arg in op.output_arg_names:
                    version = len(self._ssa_vars[output_arg])
                    # version is the length
                    self._ssa_vars[output_arg].append(VarHandle(output_arg, version))
                    self._outputs[op].add(self._ssa_vars[output_arg][-1]) # latest version
                    ready_vars.add(output_arg)
                for input_arg in op.input_arg_names:
                    self._inputs[op].add(self._ssa_vars[input_arg][-1]) # latest version
            else:
                op_queue.append(op)

    def _use_def_chain(self):
        for i in range(len(self._ops)):
            op = self._ops[i]
            self._uses[i].update(op.output_arg_names)
            self._defs[i].update(op.output_arg_names)

    def build_graph(self):
        connections = [(i, i+1) for i in range(len(self._ops) - 1)]
        for node1, node2 in connections:
            self._preds[node2].append(node1)
            self._succs[node1].append(node2)
        self._use_def_chain()
        self._find_dominators()
        self._to_ssa_graph(self._ops)

    def _up_and_mark(self, live_vars, var):
        """
        explore all paths from variable's use to its def
        """
        if var in live_vars: #Killed in the block, stop
            return
        if var in self._doms:
            return

        live_out = set()
        for p in self._preds:
            live_out = live_out[p] | set(var)
            self._up_and_mark(live_out, var)

    def liveness_range(self):
        live_out = defaultdict(set)
        for v in self._doms:
            live_out[v] = live_out[v] | self._defs[v]
            self._up_and_mark(live_out, v)
        for defs in self._defs:
            for v in defs:
                self._up_and_mark(live_out, v)

    def _build_graph(self, program):
        block = program.block(0)
        op_queue = block.ops()
        for op in block.ops():
            for output_arg_name in op.output_arg_names:
                var_handle = VarHandle(output_arg_name)
                self._vars[output_arg_name].append()

        while len(op_queue) > 0:
            op = op_queue.pop(0)
            op_handle = OpHandle(op)
            # op in startup program
            if len(op.input_arg_names) == 0:
                for output_arg in op.output_arg_names:
                    var_handle = VarHandle(output_arg)
                    var_handle.set_handle(block[output_arg], op)
                    assert output_arg not in self._vars, "output name exsits in %s"%(output_arg)
                    self._vars[output_arg] = var_handle
            for output_arg in op.output_arg_names:
                var_handle = VarHandle(output_arg)
                var_handle.set_handle(block[output_arg], op)
                assert output_arg not in self._vars, "output name exsits in %s"%(output_arg)
                self._vars[output_arg] = var_handle

            for input_arg in op.input_arg_names:
                assert input_arg in self._vars, "input name not exsits in %s"%(input_arg)
                op_handle.inputs.append(self._vars[input_arg])

        for op in self._block.ops():
            op_handle = OpHandle(op)
            if len(op.input_arg_names) == 0:
                var_handle = VarHandle()
            for input_arg in op.input_arg_names:
                if input_arg not in self._vars:
                    pass


class GraphChecker(object):
    def __init__(self, main_program, startup_program=None):
        pass

    def check_input_output(self, program):
        output_vars = set()
        input_vars = set()
        for op in program.block(0).ops:
            for output_arg in op.output_arg_names:
                output_vars.add(output_arg)
            for input_arg in op.input_arg_names:
                input_vars.add(input_arg)
        all_vars = set(program.block(0).vars)
        return all_vars - input_vars

def GenTestProgram():
    program = Program()
    startup = Program()
    with program_guard(program, startup_program=startup):
        x = layers.data(name='x', shape=[13], dtype='float32')
        y_predict = layers.fc(input=x, size=1, act=None)
        y = layers.data(name='y', shape=[1], dtype='float32')
        cost = layers.square_error_cost(input=y_predict, label=y)
        avg_cost = layers.mean(cost)
        opt = optimizer.SGD(learning_rate=0.001)
        opt = opt.minimize(avg_cost)
    return program

def GenTestProgramWithSubBlock():
    program = Program()
    startup = Program()
    with program_guard(program, startup_program=startup):
        data = layers.data(name='X', shape=[1], dtype='float32')
        data.stop_gradient = False
        cond = layers.ConditionalBlock(inputs=[data])
        out = layers.create_tensor(dtype='float32')
        with cond.block():
            hidden = layers.fc(input=data, size=10)
            layers.assign(hidden, out)
    return program

def main():
    ops = GenTestProgram()
    ssa = SSAGraph()
    ssa._run_graph(ops)

def test_cfg():
    program = GenTestProgramWithSubBlock()
    print(program)
    result_program = memory_optimize(program, print_log=True)


if __name__ == "__main__":
    test_cfg()
