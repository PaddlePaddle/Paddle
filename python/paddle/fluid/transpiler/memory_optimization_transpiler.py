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

from __future__ import print_function

import six
import sys
from collections import defaultdict, MutableSet
from .. import core
from ... import compat as cpt
from ..framework import Program, default_main_program, Parameter, Variable, core
from ..backward import _rename_arg_
from functools import reduce
from six.moves import range

dtype_to_size = {
    core.VarDesc.VarType.FP16: 2,
    core.VarDesc.VarType.FP32: 4,
    core.VarDesc.VarType.FP64: 8,
    core.VarDesc.VarType.INT16: 2,
    core.VarDesc.VarType.INT32: 4,
    core.VarDesc.VarType.INT64: 8,
    core.VarDesc.VarType.BOOL: 1,
    core.VarDesc.VarType.UINT8: 1,
}

SUB_BLOCK_OPS = [
    "while", "while_grad", "conditional_block", "conditional_block_grad"
]

SUB_BLOCK_PAIR = [("while", "while_grad"),
                  ("conditional_block", "conditional_block_grad")]

PRINT_LOG = False
FLAGS_memory_optimize = ""


class OrderedSet(MutableSet):
    def __init__(self, iterable=None):
        self.end = end = []
        end += [None, end, end]  # sentinel node for doubly linked list
        self.map = {}  # key --> [key, prev, next]
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]

    def update(self, other):
        for e in other:
            self.add(e)

    def discard(self, key):
        if key in self.map:
            key, prev, next = self.map.pop(key)
            prev[2] = next
            next[1] = prev

    def remove(self, key):
        self.discard(key)

    def __iter__(self):
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def pop(self, last=True):
        if not self:
            raise KeyError('set is empty')
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__, )
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)


class ControlFlowGraph(object):
    def __init__(self, program, ops, forward_num, skip_opt):
        self._program = program
        self._ops = ops
        self._forward_num = forward_num
        self._successors = defaultdict(OrderedSet)
        self._presuccessors = defaultdict(OrderedSet)
        self._uses = defaultdict(OrderedSet)
        self._defs = defaultdict(OrderedSet)
        self._live_in = defaultdict(OrderedSet)
        self._live_out = defaultdict(OrderedSet)

        self._skip_opt = skip_opt
        self.pool = []

    def _add_connections(self, connections):
        """Populates _successors and _presuccessors for two neighbor nodes."""
        for node1, node2 in connections:
            self._add(node1, node2)

    def _add(self, node1, node2):
        self._successors[node1].add(node2)
        self._presuccessors[node2].add(node1)

    # TODO(panyx0718): We need to have a unified way of building intermediate
    # representation.
    def _build_graph(self):
        """Build a graph based on op sequence.
        """
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
                self._live_in[i].add(new_name)
            if old_name in self._live_out[i]:
                self._live_out[i].remove(old_name)
                self._live_out[i].add(new_name)

    def _dataflow_analyze(self):
        self._build_graph()
        live_in = defaultdict(set)
        worklist = list(range(len(self._ops) - 1, -1, -1))
        while worklist:
            i = worklist.pop(0)
            live_in[i] = set(self._live_in[i])
            for s in self._successors[i]:
                self._live_out[i] |= self._live_in[s]
            self._live_in[i] = self._uses[i] | (
                self._live_out[i] - self._defs[i])
            if live_in[i] != set(self._live_in[i]):
                for d in self._presuccessors[i]:
                    worklist.append(d)

    def _fill_pool(self, i, is_forward):
        def comparator(x, cache):
            x_shape = x[1]
            cache_shape = cache[1]
            x_size = abs(reduce(lambda x, y: x * y, x_shape))
            cache_size = abs(reduce(lambda x, y: x * y, cache_shape))
            if (x_shape[0] == -1 and cache_shape[0] == -1) or \
               (x_shape[0] != -1 and cache_shape[0] != -1) :
                return x_size <= cache_size
            else:
                return False

        def find_var_in_block(x):
            known_vars = set()
            for op in self._ops:
                known_vars.update(op.output_arg_names())
            return x in known_vars

        block_desc = self._ops[i].block()
        in_diff, _ = self._get_diff(self._live_in[i], self._live_out[i])
        # NOTE: must sort the in_diff set for cases that get different cache var.
        # FIXME(typhoonzero): maybe use a "sorted set" is better than this.
        can_optimize = [
            x for x in sorted(in_diff)
            if self._check_var_validity(block_desc, x, is_forward)
        ]
        if can_optimize:
            for var_name in can_optimize:
                cache = (var_name, self._find_var(block_desc, var_name,
                                                  is_forward).shape())
                if cache not in self.pool and find_var_in_block(var_name):
                    i = 0
                    while i < len(self.pool):
                        mycache = self.pool[i]
                        mysize = mycache[1][0]
                        cache_size = cache[1][0]
                        if (mysize == -1 and cache_size == -1) or \
                           (mysize != -1 and cache_size != -1):
                            if comparator(mycache, cache):
                                i += 1
                            else:
                                break
                        elif mysize == -1 and cache_size != -1:
                            i += 1
                        elif mysize != -1 and cache_size == -1:
                            break
                    self.pool.insert(i, cache)

    def _get_diff(self, a, b):
        u = a & b
        return a - u, b - u

    def _has_var(self, block_desc, var_name, is_forward):
        if is_forward:
            return block_desc.has_var(cpt.to_bytes(var_name))
        else:
            return block_desc.has_var_recursive(cpt.to_bytes(var_name))

    def _find_var(self, block_desc, var_name, is_forward):
        if is_forward:
            return block_desc.find_var(cpt.to_bytes(var_name))
        else:
            return block_desc.find_var_recursive(cpt.to_bytes(var_name))

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

    # TODO(panyx0718): This needs to be less hacky. It seems memory optimization
    # doesn't consider vars copied between cpu and gpu.
    def _update_skip_opt_set(self):
        for i in range(self.op_size):
            op = self._ops[i]
            if op.has_attr("force_cpu") and op.attr("force_cpu") == True:
                self._skip_opt.update(op.output_arg_names())

    def release_memory(self, skip_opt_set=None):
        self._dataflow_analyze()
        self._update_skip_opt_set()
        if skip_opt_set:
            self._skip_opt.update(skip_opt_set)
        fwd_id = 0
        bwd_id = 0
        for i in range(self.op_size):
            op = self._ops[i]
            if op.type() in SUB_BLOCK_OPS:
                continue
            block_desc = op.block()
            is_forward = i < self._forward_num
            in_diff, out_diff = self._get_diff(self._live_in[i],
                                               self._live_out[i])
            can_optimize = [
                x for x in in_diff
                if self._check_var_validity(block_desc, x, is_forward)
            ]
            if can_optimize:
                index = i + fwd_id + 1 if is_forward else i - self._forward_num + bwd_id + 1
                delete_op = block_desc._insert_op(index)
                delete_op.set_type("delete_var")
                delete_op.set_input("X", can_optimize)
                if is_forward:
                    fwd_id += 1
                else:
                    bwd_id += 1

    def memory_optimize(self, skip_opt_set=None, level=0):
        def compare_shape(x_shape, cache_shape, opt_level):
            if opt_level == 0:
                return x_shape == cache_shape
            elif opt_level == 1:
                if (x_shape[0] == -1) ^ (cache_shape[0] == -1):
                    return False
                x_size = abs(reduce(lambda x, y: x * y, x_shape))
                cache_size = abs(reduce(lambda x, y: x * y, cache_shape))
                if x_size <= cache_size:
                    return True
            else:
                raise ValueError("only support opt_level 0 or 1.")
            return False

        self._dataflow_analyze()
        self._update_skip_opt_set()
        # update skip set to meet users' demand
        if skip_opt_set:
            self._skip_opt.update(skip_opt_set)
        counter = 0
        for i in range(self.op_size):
            op = self._ops[i]
            if op.type() in SUB_BLOCK_OPS:
                continue
            block_desc = op.block()
            is_forward = i < self._forward_num
            if self.pool:
                # NOTE: must sort the in_diff set for cases that get different cache var.
                defs_can_optimize = [
                    x for x in self._defs[i]
                    if self._check_var_validity(block_desc, x, is_forward)
                ]
                out_pair = [
                    (x, self._find_var(block_desc, x, is_forward).shape())
                    for x in defs_can_optimize
                ]
                for x, x_shape in out_pair:
                    # If x is both in uses and defs, it can not be optimized!
                    if x in self._uses[i]:
                        continue
                    if x == FLAGS_memory_optimize:
                        print("start match var ", x, " of op ", op.type())
                        print(self.pool)
                    for index, cache_pair in enumerate(self.pool):
                        cache_var = cache_pair[0]
                        cache_shape = cache_pair[1]
                        if not self._has_var(block_desc, cache_var, is_forward):
                            if PRINT_LOG:
                                print("cache %s not exists!" %
                                      (cpt.to_text(cache_var)))
                            continue
                        if x == cache_var:
                            if PRINT_LOG:
                                print("x : ", cpt.to_text(x), " cache : ",
                                      cpt.to_text(cache_var), " is same var!")
                            break

                        x_dtype = self._find_var(block_desc, x,
                                                 is_forward).dtype()
                        cache_dtype = self._find_var(block_desc, cache_var,
                                                     is_forward).dtype()
                        if x_dtype != cache_dtype:
                            if PRINT_LOG:
                                print("x_dtype and cache_dtype are different")
                            continue

                        if not compare_shape(x_shape, cache_shape, level):
                            continue
                        # TODO(qijun): dtype_to_size[x_dtype] and dtype_to_size[cache_dtype]
                        if PRINT_LOG:
                            print(
                                ("!!! %d,  %s => %s, cache idx %d, pool size %d"
                                 % (counter, x + str(x_shape),
                                    cache_var + str(cache_shape), index,
                                    len(self.pool))))
                            counter += 1
                        self.pool.pop(index)
                        # Rename the var to the cache var already with
                        # memory allocated in order to reuse the memory.
                        _rename_arg_(self._ops, x, cache_var, begin_idx=i)
                        self._program.block(block_desc.id).var(cpt.to_text(
                            x)).desc = self._find_var(block_desc, cache_var,
                                                      is_forward)
                        self._program.block(block_desc.id).vars[cpt.to_text(x)] = \
                            Variable(self._program.block(block_desc.id), name=cpt.to_text(x))
                        self._update_graph(x, cache_var, begin_idx=i)
                        break
            self._fill_pool(i, is_forward)


def _process_sub_block_pair(pdesc, sub_block_pair):
    """Creates a list of tuple each of which tracks info of a subblock.

      Note: this function doesn't handle nested subblocks yet.
      TODO(panyx0718): assert if case nested subblocks happen.

    :param pdesc: ProgramDesc.
    :param sub_block_pair: A list op pairs. Each op pair is the forward
        op and backward op. The ops in the list are special that they contain
        a subblock of ops.
    :return: A list of tuples, each tuple is (all ops in a subblock pair
        including forward and backward, number of forward ops,
        all output args names of the ops in the subblock pairs).
    """
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
            sub_op_output.update(sub_op_dict[fwd_id].input_arg_names())
            sub_op_output.update(sub_op_dict[grad_id].input_arg_names())
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
            sub_op_output.update(sub_op_dict[fwd_id].input_arg_names())
            ops_list.append((sub_block_ops, sub_block_op_size, sub_op_output))
    return ops_list


def _get_cfgs(input_program):
    """Process each block and create ControlFlowGraph for each of them.

    :param input_program: Program object.
    :return: A list of ControlFlowGraph, each corresponds to a block.
    """
    ops_list = []
    pdesc = input_program._get_desc()
    block_desc = pdesc.block(0)
    op_size = block_desc.op_size()

    # Only process one level of nested subblock.
    ops_list.extend(_process_sub_block_pair(pdesc, SUB_BLOCK_PAIR))

    skip_opt_set = set()
    for _, _, skip_opt in ops_list:
        skip_opt_set.update(skip_opt)

    # Get global block ops
    ops_list.insert(
        0, ([block_desc.op(i) for i in range(op_size)], op_size, skip_opt_set))
    cfgs = [
        ControlFlowGraph(input_program, ops, forward_num, skip_opt)
        for ops, forward_num, skip_opt in ops_list
    ]
    return cfgs


def _is_opt_role_op(op):
    op_maker = core.op_proto_and_checker_maker
    optimize_role = core.op_proto_and_checker_maker.OpRole.Optimize
    if op_maker.kOpRoleAttrName() in op.attr_names and \
            int(op.all_attrs()[op_maker.kOpRoleAttrName()]) == int(optimize_role):
        return True


def memory_optimize(input_program,
                    skip_opt_set=None,
                    print_log=False,
                    level=0,
                    skip_grads=False):
    """Optimize memory by reusing var memory.

      Note: it doesn't not support subblock nested in subblock.

    Args:
        input_program(str): Input Program
        skip_opt_set(set): vars wil be skipped in memory optimze
        print_log(bool): whether to print debug log.
        level(int): If level=0, reuse if the shape is completely equal, o
    Returns:
        None

    Examples:
        .. code-block:: python
            main_prog = fluid.Program()
            startup_prog = fluid.Program()

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)

            exe.run(startup_prog)
            fluid.memory_optimize(main_prog)
    """
    sys.stderr.write('memory_optimize is deprecated. '
                     'Use CompiledProgram and Executor\n')

    def to_name_str(var):
        if isinstance(var, Variable):
            return var.desc.name()
        elif isinstance(var, str):
            return var
        elif isinstance(var, six.string_types):
            return str(var)
        else:
            raise TypeError(str(var) + " should be Variable or str")

    if level != 0 and level != 1:
        raise ValueError("only support opt_level 0 or 1.")
    if skip_opt_set is not None:
        if isinstance(skip_opt_set, set) or isinstance(skip_opt_set, list):
            skip_opt_set = set(skip_opt_set)
        else:
            raise ValueError("only support skip_opt_set as set.")
    global PRINT_LOG
    PRINT_LOG = print_log
    if skip_grads:
        grad_set = set()
        OP_ROLE_VAR = core.op_proto_and_checker_maker.kOpRoleVarAttrName()
        for op in input_program.global_block().ops:
            if _is_opt_role_op(op):
                if op.attr(OP_ROLE_VAR):
                    grad_name = op.attr(OP_ROLE_VAR)[1]
                    grad_set.add(grad_name)
        if not skip_opt_set:
            skip_opt_set = grad_set
        else:
            skip_opt_set.update(grad_set)
    if skip_opt_set is not None:
        skip_opt_set = set(map(to_name_str, skip_opt_set))
    cfgs = _get_cfgs(input_program)
    input_program._is_mem_optimized = True
    for cfg in cfgs:
        cfg.memory_optimize(skip_opt_set=skip_opt_set, level=level)


def release_memory(input_program, skip_opt_set=None):
    """
    Modify the input program and insert :code:`delete_op` to early drop not used
    variables. The modification will be performed inplace.

    Notes: This is an experimental API and could be removed in next few
    releases. Users should not use this API.

    Args:
        input_program(Program): The program will be inserted :code:`delete_op`.
        skip_opt_set(set): vars wil be skipped in memory optimze
    Returns:
        None

    Examples:
        .. code-block:: python
            main_prog = fluid.Program()
            startup_prog = fluid.Program()

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)

            exe.run(startup_prog)
            fluid.release_memory(main_prog)
    """
    cfgs = _get_cfgs(input_program)
    input_program._is_mem_optimized = True
    for cfg in cfgs:
        cfg.release_memory(skip_opt_set=skip_opt_set)
