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
import operator

import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.layers as layers
import paddle.fluid.optimizer as optimizer
from paddle.fluid.framework import Program, program_guard
from paddle.fluid.transpiler import memory_optimize
from paddle.fluid.framework import Program, default_main_program, Parameter, Variable
from paddle.fluid.backward import _rename_arg_, append_backward
"""
MemoryTranspiler implement the memory reuse strategy.
It mainly composed by two type optimize **op inplace computation** and **non-lived variable reuse**.

It follow the steps:
1. create memory plan for each Block.
2. for each memory plan:
  2.1 do topology sort, compute the dependency for each var, compute the variable liveness range.
  2.2 create cache pool, put non-lived variable into it, for new created var, if hit cache, then reuse.
  2.3 for each op, if it output marked with reuse and make sure the op is the last referenced op. do inplace.
done.
"""


class Graph(object):
    """
    A basic graph, each op as graph nodes.
    :param ops: a list of Operator instance in a Block.
    :return:
    """

    def __init__(self, ops):
        self._ops = ops
        self._ops_dep_var = defaultdict(
            list)  # var name x -> op set deps on var x
        self._vars = defaultdict(set)
        self._link_dependency()

    def _link_dependency(self):
        for op in self._ops:
            for input_arg in op.input_arg_names:
                self._ops_dep_var[input_arg].append(op)

    def get_ops_dep_var(self, var_name):
        return self._ops_dep_var[var_name]

    def _remove_circle(self, ops):
        no_circle_op_role = set([
            int(core.op_proto_and_checker_maker.OpRole.Forward),
            int(core.op_proto_and_checker_maker.OpRole.Backward),
            int(core.op_proto_and_checker_maker.OpRole.Loss),
        ])
        dag_ops = []
        for op in ops:
            if op.attr("op_role") in no_circle_op_role:
                dag_ops.append(op)
        return dag_ops

    def _get_circle(self, ops):
        circle_op_role = set([
            int(core.op_proto_and_checker_maker.OpRole.RPC),
            int(core.op_proto_and_checker_maker.OpRole.Optimize),
        ])
        dag_ops = []
        for op in ops:
            if op.attr("op_role") in circle_op_role:
                dag_ops.append(op)
        return dag_ops

    def _collect_dangling_vars(self):
        input_vars = set()
        output_vars = set()
        ops = self._remove_circle(self._ops)
        for op in ops:
            for input_arg in op.input_arg_names:
                input_vars.add(input_arg)
            for output_arg in op.output_arg_names:
                output_vars.add(output_arg)
        return input_vars - output_vars

    def topology_sort(self):
        def ready(op, var_set):
            for input_arg in op.input_arg_names:
                if input_arg not in var_set:
                    return False
            return True

        outputs = self._collect_dangling_vars()
        no_circle_ops = self._remove_circle(self._ops)
        ops = []
        while no_circle_ops:
            op = no_circle_ops.pop(0)
            if ready(op, outputs):
                ops.append(op)
                outputs.update(op.output_arg_names)
            else:
                no_circle_ops.append(op)
        circle_ops = self._get_circle(self._ops)
        ops.extend(circle_ops)
        return ops


class ControlFlowGraph(object):
    """
    The CFG is contructed for compute the variable liveness range.
    live_in, live_out mark every variable's live timeline.
    :param ops: a list of Operator instance in a Block,
               the order of ops matters.
    :return:
    """

    def __init__(self, ops):
        self._ops = ops
        self._analysis_finish = False
        self._successors = defaultdict(list)  # the succs before current op
        self._predecessors = defaultdict(list)  # the preds before current op
        self._live_in = defaultdict(
            set)  # the live_in variables before current op
        self._live_out = defaultdict(set)
        self._uses = defaultdict(set)  # variables used in op
        self._defs = defaultdict(set)  # new generated variables

    def _update_use_def_chain(self):
        for i in range(len(self._ops)):
            self._uses[i].update(self._ops[i].input_arg_names)
            self._defs[i].update(self._ops[i].output_arg_names)

    def _connect(self, connections):
        for node1, node2 in connections:
            self._successors[node1].append(node2)
            self._predecessors[node2].append(node1)

    def _build_graph(self):
        """
        NOTE(dzh):
        By defination, the variable v is live on edge e if there is a node n
        in the CFG that uses it and a directed path from
        e to n passing through no def.
        So, its concrete form should be the variable dependency graph. However, the liveness only
        propagate from live-out to live-in, so just link the neighbor works.
        we use connect at every node simply instead of a var-by-var DFS.
        """
        connections = [(i, i + 1) for i in range(len(self._ops) - 1)]
        self._update_use_def_chain()
        self._connect(connections)

    def compute_liveness_range(self):
        """
        compute the liveness of for each op.
        the worklist algorithm refer to
        http://www.cs.cornell.edu/courses/cs4120/2013fa/lectures/lec26-fa13.pdf
        """

        self._build_graph()
        live_in = defaultdict(set)
        worklist = range(len(self._ops) - 1, -1, -1)
        while worklist:
            i = worklist.pop(0)
            live_in[i] = set(self._live_in[i])
            for s in self._successors[i]:
                self._live_out[i] |= self._live_in[s]
            self._live_in[i] = self._uses[i] | (
                self._live_out[i] - self._defs[i])
            if live_in[i] != self._live_in[i]:
                for d in self._predecessors[i]:
                    worklist.append(d)
        self._analysis_finish = True

    def rename(self, old_name, new_name, begin_idx):
        for i in range(begin_idx, len(self._ops)):
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

    def live_in(self, idx):
        if not self._analysis_finish:
            raise RuntimeError("Not finish liveness computing")
        if idx >= len(self._ops) or idx < 0:
            raise IndexError("Out of range.")
        return self._live_in[idx]

    def live_out(self, idx):
        if not self._analysis_finish:
            raise RuntimeError("Not finish liveness computing")
        if idx >= len(self._ops) or idx < 0:
            raise IndexError("Out of range.")
        return self._live_out[idx]

    def get_def(self, idx):
        return self._defs[idx]

    def get_use(self, idx):
        return self._uses[idx]


class MemoryBlock(object):
    """
    MemoryBlock is the memory beneath variable placeholder.
    its attribute is used in cache matching.

    :param name: variable name
    :param dtype: variable data type
    :param shape: variable data shape
    :param desc: var_desc, for the target var candidate.
    :param place: place, memory stays place. It will be used
                 when "force_cpu" is True.
    """

    def __init__(self, name, dtype, shape, desc, place=None):
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.place = place
        self.var_desc = desc

    def __str__(self):
        return "[%s,%s,%s,%s]" % (self.name, str(self.dtype), str(self.shape),
                                  str(self.place))

    def get_size(self):
        """
        if shape is [-1, 1, ...], then we only compare the rest part.
        """
        # return abs(reduce(lambda x, y: x * y, self.shape)) * core.size_of_type(
        #     self.dtype)
        return abs(reduce(lambda x, y: x * y, self.shape))

    def _compare_attr(self, rhs, functor):
        if self.dtype != rhs.dtype or self.place != rhs.place:
            return False
        if not functor(self.get_size(), rhs.get_size()):
            return False
        return True

    def GE(self, rhs):
        return self._compare_attr(rhs, operator.ge)

    def EQ(self, rhs):
        return self._compare_attr(rhs, operator.eq) and self.shape == rhs.shape


class MemoryPlan(object):
    """
    MemoryPlan optimize the memory reuse for each block.
    It takes a fresh Python Block and generate a optimized BlockDesc.

    :param block: Python Block instance
    :param logging: print log info
    """

    def __init__(self, block, logging=False):
        self.logging = logging
        self._block = block  # The BlockDesc beneath it will be optimized.
        self._memory_pool = []  # cache pool
        self._ir = None  # ControlFlowGraph
        self._graph = None  # Dependency graph
        self._ops = None  # sorted ops

    def _liveness_plan(self):
        ops = self._block.ops
        graph = Graph(ops)
        graph.topology_sort()
        cfg = ControlFlowGraph(ops)
        cfg.compute_liveness_range()
        self._graph = graph
        self._ops = ops
        self._vars = self._block.vars
        self._ir = cfg

    def _is_memory_block(self, var_name):
        if var_name not in self._vars:
            return False
        x = self._var_names[var_name]
        if x.persistable or x.shape == None:
            return False
        # NOTE(dzh): assert x.type() is seperated intendly, because if it not hold a Tensor, it
        # will not have the attribute of type.
        if x.type() != core.VarDesc.VarType.LOD_TENSOR:
            return False
        return True

    def _create_memory_block(self, op, var_name):
        var = self._vars[var_name]
        place = None
        if op.has_attr("force_cpu") and op.attr("force_cpu") == True:
            place = fluid.CPUPlace()
        return MemoryBlock(var.name, var.dtype, var.shape, var.desc, place)

    def _find_memory_block(self, memory_block):
        found = False
        for idx, memory in enumerate(self._memory_pool):
            if memory.EQ(memory_block):
                found = True
                break
        if found:
            self._memory_pool.pop(idx)
            return idx, memory
        else:
            return -1, None

    def _reuse_cache(self, cache, op, var_name, idx):
        op_descs = [op.desc for op in self._ops]
        _rename_arg_(op_descs, var_name, cache.name, begin_idx=idx)
        self._block.var(var_name).desc.name = cache.name
        self._ir.rename(var_name, cache.name, begin_idx=idx)

    def _reuse_inplace(self, op, var_name, idx):
        reuse_var = op.attr("reuse")
        ops_dep_var = self._graph.get_ops_dep_var(reuse_var)
        """
        only the last referenced op can apply inplace to the variable.
        For example, x is the variable. x_a want to reuse x in op_a, and x_b want to reuse x in op_b,
        if all of these two op use inplace, then the second op will get wrong.
        So, we build the dependency graph, and only let the last op apply inplace.
        """
        if ops_dep_var[-1] != op:
            return
        op_descs = [op.desc for op in self._ops]
        _rename_arg_(op_descs, var_name, reuse_var, begin_idx=idx)
        self._block.var(var_name).desc.name = reuse_var
        self._ir.rename(var_name, reuse_var, begin_idx=idx)

    def apply(self):
        self._liveness_plan()
        for idx, op in enumerate(self._ops):
            if op.has_sub_block():
                continue
            block_desc = op.block
            var_defs = filter(lambda x: self._is_memory_block,
                              self._ir.get_def(idx))
            for var_name in var_defs:
                if op.has_attr("reuse"):
                    self._reuse_inplace(op, var_name, idx)
                if var_name not in self._ir.get_use(idx):
                    memory = self._create_memory_block(op, var_name)
                    index, cache = self._find_memory_block(memory)
                    if cache:
                        if self.logging:
                            print(("Hit Cache !!!! cache pool index "
                                   "is %d, var is %s, "
                                   "cached var is %s.") % (index, str(memory),
                                                           str(cache)))
                        self._reuse_cache(cache, op, var_name, idx)
                        break
            live_in = self._ir.live_in(idx)
            live_out = self._ir.live_out(idx)
            caches_candidate = live_in - (live_in & live_out)
            caches = filter(lambda x: self._is_memory_block, caches_candidate)
            for cache in caches:
                self._memory_pool.append(self._create_memory_block(op, cache))


class MemoryTranspiler(object):
    """
    MemoryTranspiler transpile the program memory.
    It take a instance of Python Program, analysis the non-lived variable then do the re-use,
    also the in-place op computations.

    :param program: Python Program instance
    :param logging: print log info
    """

    def __init__(self, program, logging=False):
        if not isinstance(program, Program):
            raise ValueError("expect argument Program, but get %s",
                             type(program))
        self.logging = logging
        self._program = program
        self._plans = []

    def _create_plans(self):
        block = self._program.block(0)
        sub_blocks = [block]
        for op in block.ops:
            if op.has_sub_block():
                block_id = op.attr("sub_block").id
                sub_blocks.append(self._program.block(block_id))
        print(sub_blocks)
        for block in sub_blocks:
            self._plans.append(MemoryPlan(block, self.logging))

    def transpile(self):
        self._create_plans()
        for plan in self._plans:
            plan.apply()


def GenProgram1():
    program = Program()
    with program_guard(program, startup_program=Program()):
        x = layers.data(name='x', shape=[13], dtype='float32')
        y_predict = layers.fc(input=x, size=1, act=None)
        y = layers.data(name='y', shape=[1], dtype='float32')
        cost = layers.square_error_cost(input=y_predict, label=y)
        avg_cost = layers.mean(cost)
        opt = optimizer.SGD(learning_rate=0.001)
        opt = opt.minimize(avg_cost)
    return program


def GenProgram2():
    program = Program()
    with program_guard(program, startup_program=Program()):
        data = layers.data(name='X', shape=[1], dtype='float32')
        data.stop_gradient = False
        cond = layers.ConditionalBlock(inputs=[data])
        out = layers.create_tensor(dtype='float32')
        with cond.block():
            hidden = layers.fc(input=data, size=10)
            layers.assign(hidden, out)
        loss = layers.mean(out)
        append_backward(loss=loss)
    return program


if __name__ == "__main__":
    program = GenProgram2()
    # program = GenProgram1()
    # print program
    MemoryTranspiler(program).transpile()
    # plan = MemoryPlan(program, logging=True)
    # plan.apply()
