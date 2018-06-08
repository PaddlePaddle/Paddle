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

from pprint import pprint
from collections import defaultdict, OrderedDict
import paddle.fluid.core as core

import paddle.fluid.layers as layers
import paddle.fluid.optimizer as optimizer
from paddle.fluid.framework import Program, program_guard, Variable, Operator
from memory_optimization_transpiler import memory_optimize, release_memory

dtype_to_size = {
    core.VarDesc.VarType.FP16: 2,
    core.VarDesc.VarType.FP32: 4,
    core.VarDesc.VarType.FP64: 8,
    core.VarDesc.VarType.INT16: 2,
    core.VarDesc.VarType.INT32: 4,
    core.VarDesc.VarType.INT64: 8,
    core.VarDesc.VarType.BOOL: 1,
    # core.VarDesc.VarType.UINT8: 1,
}



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

    # def _transform_ssa_graph(self, ops):
    #     for op in ops:
    def _no_output_vars(self, ops):
        inputs = set()
        outputs = set()
        for op in ops:
            for input_arg in op.input_arg_names:
                inputs.add(input_arg)
            for output_arg in op.output_arg_names:
                outputs.add(output_arg)
        return inputs - outputs

    def _run_graph(self, ops):
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
        # ready_vars = set()
        op_queue = list(ops)
        while op_queue:
            op = op_queue.pop(0)
            # print(op.type)
            # print(ready_ops)
            # print(ready_vars)
            # r_ops = [r_op.type for r_op in ready_ops]
            # print(r_ops)
            # print(len(ready_vars))
            if can_run(op, ready_vars):
                # print(op.type)
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
        # print(self._ssa_vars)
        max_len = max([len(var) for var in self._ssa_vars])
        print(len(self._ssa_vars))
        print(len(self._inputs))
        print("max_len : ", max_len)


    def _build_graph2(self, block):
        for op in block.ops:
            for output_arg_name in op.output_arg_names:
                version = len(self._vars[output_arg_name])
                self._vars[output_arg_name].append(version)
        all_vars = list()
        input_vars = list()
        output_vars = list()
        for v in block.vars:
            all_vars.append(v)
        for op in block.ops:
            for arg_name in op.input_arg_names:
                input_vars.append(arg_name)
            for arg_name in op.output_arg_names:
                output_vars.append(arg_name)

        all_vars.sort()
        input_vars.sort()
        output_vars.sort()
        pprint(all_vars)
        pprint(input_vars)
        pprint(output_vars)
        pprint(self._vars)

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
                    # self._vars[input_arg] =
                    pass

    def build_graph(self):
        """
        build ssa graph
        """

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
        # print all_vars - output_vars
        print all_vars - input_vars

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
    # print startup.block(0).ops
    return program
    # return program.block(0).ops + startup.block(0).ops

def main():
    ops = GenTestProgram()
    # print(program)
    # ssa = SSAGraph(program)
    ssa = SSAGraph()
    ssa._run_graph(ops)
    # gc = GraphChecker(program)
    # gc.check_input_output(program)
    # print(str(program))
    # print program.block(0).ops
    # check_program(program)

def test_cfg():
    program = GenTestProgram()
    result_program = memory_optimize(program, print_log=True)
    # print program.block(0).ops


if __name__ == "__main__":
    # main()
    test_cfg()
    # result_program = memory_optimize(program)
    # print("after optimization")
    # print(str(program))
