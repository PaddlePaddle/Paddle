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

from paddle.fluid import framework as framework
from . import core
import collections
import copy
import six
import logging
from .. import compat as cpt
from . import unique_name
from . import log_helper
import paddle.fluid
__all__ = [
    'append_backward',
    'gradients',
]

_logger = log_helper.get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


class ProgramStats(object):
    def __init__(self, block, ops):
        self.block = block
        self.ops = ops
        self.op_deps = {}  # op-> in_ops, out_ops
        self.var_op_deps = {}  # var as input op, var as output op

    def get_input_nodes(self):
        input_names = []
        for name in self.var_op_deps:
            if len(self.var_op_deps[name]["var_as_output_ops"]) == 0 and \
               len(self.var_op_deps[name]["var_as_input_ops"]) > 0:
                if self.block.var(name).persistable:
                    continue
                input_names.append(name)
        for op in self.ops:
            if op.desc.type() == "read":
                input_names.extend(op.desc.output_arg_names())
        return input_names

    def get_reserved_vars(self):
        var_name = []
        for op in self.ops:
            if op.desc.type() == "dropout":
                var_name.extend(op.desc.output_arg_names())
        return var_name

    def get_out_of_subgraph_vars(self, begin_op_idx, end_op_idx):
        var_name = []
        for i in range(begin_op_idx, end_op_idx, 1):
            for name in self.ops[i].desc.output_arg_names():
                if name in self.var_op_deps:
                    for idx in self.var_op_deps[name]["var_as_input_ops"]:
                        if idx >= end_op_idx:
                            var_name.append(name)
        return var_name

    def is_subgraph(self, var_group1, var_group2):
        # should traverse from var_group1 to var_group2
        # max op idx in var_group2
        # min op idx in var_group1
        min_op_idx = len(self.ops)
        max_op_idx = -1
        for name in var_group1:
            if name not in self.var_op_deps:
                return False, min_op_idx, max_op_idx
        for name in var_group2:
            if name not in self.var_op_deps:
                return False, min_op_idx, max_op_idx
        for name in var_group1:
            op_idx = self.var_op_deps[name]["var_as_input_ops"]
            for idx in op_idx:
                min_op_idx = min(min_op_idx, idx)
        for name in var_group2:
            op_idx = self.var_op_deps[name]["var_as_output_ops"]
            for idx in op_idx:
                max_op_idx = max(max_op_idx, idx)
        if min_op_idx >= max_op_idx:
            return False, min_op_idx, max_op_idx
        return True, min_op_idx, max_op_idx

    def build_stats(self):
        for i, op in enumerate(self.ops):
            self.op_deps[i] = {"in_ops": [], "out_ops": []}
            for j, name in enumerate(op.desc.input_arg_names()):
                if name in self.var_op_deps:
                    self.op_deps[i]["in_ops"].extend(self.var_op_deps[name][
                        "var_as_output_ops"])
            for j, name in enumerate(op.desc.input_arg_names()):
                if name in self.var_op_deps:
                    self.var_op_deps[name]["var_as_input_ops"].extend([i])
                else:
                    self.var_op_deps[name] = {}
                    self.var_op_deps[name]["var_as_input_ops"] = [i]
                    self.var_op_deps[name]["var_as_output_ops"] = []

            for j, name in enumerate(op.desc.output_arg_names()):
                if name in self.var_op_deps:
                    self.var_op_deps[name]["var_as_output_ops"].extend([i])
                else:
                    self.var_op_deps[name] = {}
                    self.var_op_deps[name]["var_as_input_ops"] = []
                    self.var_op_deps[name]["var_as_output_ops"] = [i]

            for op_idx in self.op_deps[i]["in_ops"]:
                self.op_deps[op_idx]["out_ops"].extend([i])

    def sort_checkpoints(self, checkpoints_name):
        sorted_checkpoints = []
        for name in checkpoints_name:
            if name not in self.var_op_deps:
                _logger.debug(
                    "Recompute Optimizer: deleted %s from checkpoints, because it is not used in paddle program."
                    % name)
            elif self.var_op_deps[name]["var_as_output_ops"] == []:
                # input nodes
                sorted_checkpoints.append((name, -1))
            else:
                sorted_checkpoints.append(
                    (name, max(self.var_op_deps[name]["var_as_output_ops"])))
        sorted_checkpoints = sorted(sorted_checkpoints, key=lambda x: x[1])
        return [x[0] for x in sorted_checkpoints]


def _pretty_op_desc_(op_desc, prefix):
    out_s = "%s\tname:[%s]\n%s    \tinputs:[%s]\n%s    \toutputs:[%s]" % \
            (prefix + "_op", str(op_desc.type()), prefix + "_input", " ".join(op_desc.input_arg_names()),
             prefix + "_output", " ".join(op_desc.output_arg_names()))
    return out_s


def _add_needed_descs_to_block(descs, block, main_block, in_memory_vars):
    if len(descs) == 0:
        return []
    result_descs = []
    op_role_attr_name = \
            core.op_proto_and_checker_maker.kOpRoleAttrName()
    backward = core.op_proto_and_checker_maker.OpRole.Backward
    for desc in descs:
        if isinstance(desc, framework.Operator):
            desc = desc.desc
        if isinstance(desc, tuple):
            desc = desc[0]
        is_needed = False
        for name in desc.output_arg_names():
            if main_block.has_var(name) and main_block.var(name).persistable:
                continue
            if name not in in_memory_vars:
                is_needed = True
        if is_needed:
            new_op_desc = block.desc.append_op()
            new_op_desc.copy_from(desc)
            new_op_desc._set_attr(op_role_attr_name, backward)
            result_descs.append(new_op_desc)
    return result_descs


def _add_descs_to_block(descs, block):
    if len(descs) == 0:
        return []
    result_descs = []
    op_role_attr_name = \
        core.op_proto_and_checker_maker.kOpRoleAttrName()
    backward = core.op_proto_and_checker_maker.OpRole.Backward
    for desc in descs:
        if isinstance(desc, framework.Operator):
            desc = desc.desc
        if isinstance(desc, tuple):
            desc = desc[0]
        new_op_desc = block.desc.append_op()
        new_op_desc.copy_from(desc)
        new_op_desc._set_attr(op_role_attr_name, backward)
        result_descs.append(new_op_desc)
    return result_descs


def _find_loss_op_(loss):
    for op in reversed(loss.block.ops):
        assert isinstance(op, framework.Operator)
        if len(op.output_arg_names) == 1 and op.output_arg_names[
                0] == loss.name:
            loss.op = op
            break
    if loss.op is None:
        raise ValueError("loss.op is None. Should not happend")


def _rename_arg_(op_descs, old_name, new_name, begin_idx=None, end_idx=None):
    """
    Traverse all ops in op_descs[begin_idx : end_idx],
    if any op has inputs/outputs named "old_name", rename it as 'new_name'
    """
    if begin_idx is None:
        begin_idx = 0
    if end_idx is None:
        end_idx = len(op_descs)
    for i in range(begin_idx, end_idx):
        op_desc = op_descs[i]
        if isinstance(op_desc, tuple):
            op_desc = op_desc[0]
        op_desc._rename_input(old_name, new_name)
        op_desc._rename_output(old_name, new_name)


def _create_op_desc_(op_type, inputs, outputs, attrs):
    """
    Create a C++ OpDesc object with specified inputs, outputs and attributes.
    """
    op_desc = core.OpDesc()
    op_desc.set_type(op_type)
    for para, args in six.iteritems(inputs):
        op_desc.set_input(
            para,
            list(
                map(lambda arg: arg.decode() if isinstance(arg, six.binary_type) else arg,
                    args)))
    for para, args in six.iteritems(outputs):
        op_desc.set_output(
            para,
            list(
                map(lambda arg: arg.decode() if isinstance(arg, six.binary_type) else arg,
                    args)))

    op_role_attr_name = core.op_proto_and_checker_maker.kOpRoleAttrName()

    if op_role_attr_name not in attrs:
        attrs[
            op_role_attr_name] = core.op_proto_and_checker_maker.OpRole.Backward
    for name, val in six.iteritems(attrs):
        if isinstance(val, framework.Block):
            op_desc.set_block_attr(name, val.desc)
        else:
            op_desc._set_attr(name, val)
    return op_desc


def _create_loss_op_desc_(loss):
    op_desc = _create_op_desc_(
        "fill_constant", {}, {"Out": [_append_grad_suffix_(loss.name)]}, {
            "shape": [1],
            "value": 1.0,
            "dtype": loss.dtype,
            "force_cpu": False,
            core.op_proto_and_checker_maker.kOpRoleAttrName():
            int(core.op_proto_and_checker_maker.OpRole.Backward) |
            int(core.op_proto_and_checker_maker.OpRole.Loss),
        })
    return op_desc


def _infer_var_data_type_(grad_var_name, block):
    """
    Infer the data type of given grad variable
    """
    grad_var = block.desc.find_var(cpt.to_bytes(grad_var_name))
    fwd_name = _strip_grad_suffix_(grad_var_name)
    if block.desc.has_var_recursive(cpt.to_bytes(fwd_name)):
        fwd_var = block.desc.find_var_recursive(cpt.to_bytes(fwd_name))
        grad_var.set_dtype(fwd_var.dtype())
    else:
        grad_var.set_dtype(core.VarDesc.VarType.FP32)


def _all_in_set_(cands, s):
    """
    Test if all elements of 'cands' are in set 's'
    """
    if len(cands) == 0:
        return False
    for c in cands:
        if not c in s:
            return False
    return True


def _some_in_set_(cands, s):
    """
    Test if some elements of 'cands' are in set 's'
    """
    if len(cands) == 0:
        return False
    literal_set = cpt.to_text(s)
    literal_cands = cpt.to_text(cands)
    for c in literal_cands:
        if c in literal_set:
            return True
    return False


def _strip_grad_suffix_(name):
    """
    Strip the grad suffix from the given variable name
    e.g. x@GRAD ==> x
         y@GRAD@RENAME@1 ==> y
    """
    name = cpt.to_text(name)
    pos = name.find(core.grad_var_suffix())
    return name[:pos] if pos != -1 else name


def _append_grad_suffix_(name):
    """
    Append grad suffix to the given variable name
    e.g. x ==> x@GRAD
    """
    return cpt.to_text(name) + core.grad_var_suffix()


def _addup_repetitive_outputs_(op_descs):
    """
    In backward part, an variable may be the output of more than one ops.
    And one op may yield its multiple outputs to the same variable.
    In these cases, the variable should be the accumulation of all the outputs.
    `sum_op`s are added to implement the accumulate.
    """
    pending_sum_ops = []
    var_rename_count = collections.defaultdict(int)
    renamed_vars = collections.defaultdict(list)
    renamed_var_start_idx = collections.defaultdict(list)
    for idx, op_desc in enumerate(op_descs):
        for var_name in op_desc.input_arg_names():
            if "@GRAD" not in var_name:
                continue
            if len(renamed_vars[var_name]) > 1:
                pending_sum_ops.append((_create_op_desc_(
                    "sum", {"X": renamed_vars[var_name]}, {"Out": [var_name]},
                    {"use_mkldnn": False}), idx))
                renamed_vars[var_name] = [var_name]
        for param_idx, param_name in enumerate(op_desc.output_names()):
            arg_names = op_desc.output(param_name)
            for arg_idx, var_name in enumerate(arg_names):
                if "@GRAD" not in var_name:
                    continue
                #if "@RENAME@" in var_name:
                #    continue
                if var_name == core.empty_var_name(
                ) or var_name in op_desc.input_arg_names():
                    # empty variable or inplace op
                    continue
                if len(renamed_vars[var_name]) == 0:
                    # it's the first time we get the variable
                    renamed_vars[var_name] = [var_name]
                    renamed_var_start_idx[var_name] = idx
                else:
                    if len(renamed_vars[var_name]) == 1:
                        new_name = var_name + "@RENAME@" + \
                            str(var_rename_count[var_name])
                        var_rename_count[var_name] += 1
                        # rename original var_name
                        renamed_vars[var_name][0] = new_name
                        # before change: _rename_arg_(op_descs, var_name,
                        #                             new_name, 0, idx)
                        # rename arg from idx of the first appearance
                        # in backward, not always from 0
                        _rename_arg_(op_descs, var_name, new_name,
                                     renamed_var_start_idx[var_name], idx)
                        _rename_arg_(pending_sum_ops, var_name, new_name)

                        for p in op_desc.output_names()[:param_idx]:
                            p_arg_names = op_desc.output(p)
                            if var_name in p_arg_names:
                                op_desc.set_output(p, [
                                    new_name if x == var_name else x
                                    for x in p_arg_names
                                ])

                        arg_names = [
                            new_name if x == var_name else x
                            for x in arg_names[:arg_idx]
                        ] + arg_names[arg_idx:]

                    new_name = var_name + "@RENAME@" + \
                        str(var_rename_count[var_name])
                    var_rename_count[var_name] += 1
                    arg_names[arg_idx] = new_name
                    op_desc.set_output(param_name, arg_names)
                    renamed_vars[var_name].append(new_name)

    for var_name, inputs in six.iteritems(renamed_vars):
        if len(inputs) > 1:
            pending_sum_ops.append(
                (_create_op_desc_("sum", {"X": inputs}, {"Out": [var_name]},
                                  {"use_mkldnn": False}), len(op_descs)))
    # sum_op descs are sorted according to their insert position
    for p in reversed(pending_sum_ops):
        op_descs.insert(p[1], p[0])

    return op_descs


def _remove_no_grad_branch_(op_descs, no_grad_set):
    """
    Remove unnecessary grad ops
    A grad op can be removed in two cases:
        1. all outputs of the grad op are in 'no_grad_set'
        2. all grad inputs of the grad op are in 'no_grad_set'
    """

    def _op_can_be_removed_(op_desc, no_grad_set):
        out_arg_names = op_desc.output_arg_names()
        if len(out_arg_names) == 0 or _all_in_set_(out_arg_names, no_grad_set):
            return True
        if _all_in_set_([
                name for name in op_desc.input_arg_names()
                if name.find(core.grad_var_suffix()) != -1
        ], no_grad_set):
            no_grad_set.update(out_arg_names)
            return True
        return False

    # Remove ops whose outputs are all in no_grad_dict
    op_descs = [
        op_desc for op_desc in op_descs
        if not _op_can_be_removed_(op_desc, no_grad_set)
    ]
    # Insert fill_zeros_like_op
    to_insert = []
    for idx, op_desc in enumerate(op_descs):
        for arg in op_desc.input_arg_names():
            # arg is a gradient var name and arg should not have gradient
            if core.grad_var_suffix() in arg and arg in no_grad_set:
                x_in = _strip_grad_suffix_(arg)
                # the reason should be: arg can be input of another grad op
                # and the op is a not-to-remove op
                to_insert.append((_create_op_desc_(
                    "fill_zeros_like", {"X": [x_in]}, {"Out": [arg]}, {}), idx))

    list([op_descs.insert(p[1], p[0]) for p in reversed(to_insert)])

    return op_descs


def _find_not_need_ops(grad_op_descs, forward_ops, input_grad_names_set):
    """
    Pruning Program with Structural Analysis Method of Computational Graph.
    The nodes of the computational graph composed of backward OPS should be
    interconnected. If there are unconnected sub-graphs in the computational graph,
    these sub-graphs should be cut off.

    Args:
        grad_op_descs(list[core.OpDesc]): The candidate backward OpDescs.
        forward_ops(list[Operator]): The forward ops.
        input_grad_names_set(set): this set is used to store the gradients' name
            which is generated by backward ops, and input_grad_names_set can help
            to prune the unnecessary backward ops.

    Return:
        (list[core.OpDesc]): A list of OpDescs which should be pruned.
    """

    class Var(object):
        def __init__(self, var_name):
            self.var_name = var_name
            self.gen_op = None
            self.pendding_ops = []

        def set_gen_op(self, gen_op):
            assert isinstance(gen_op, Op)
            assert self.gen_op is None
            self.gen_op = gen_op

        def add_pending_op(self, op):
            assert isinstance(op, Op)
            self.pendding_ops.append(op)

    class Op(object):
        def __init__(self, op_desc):
            self.op_desc = op_desc
            self.inputs = []
            self.outputs = []

        def insert_input(self, var):
            assert isinstance(var, Var)
            self.inputs.append(var)

        def insert_output(self, var):
            assert isinstance(var, Var)
            self.outputs.append(var)

    var_versions = dict()

    def _create_node(name):
        if name not in var_versions.keys():
            var_versions[name] = [Var(name)]
        else:
            var_versions[name].append(Var(name))
        return var_versions[name][-1]

    def _create_or_get_last_version_node(name):
        if name not in var_versions.keys():
            var_versions[name] = [Var(name)]
        return var_versions[name][-1]

    def _create_op_node(op_desc):
        op_node = Op(op_desc)
        for input in op_desc.input_arg_names():
            var = _create_or_get_last_version_node(name=input)
            var.add_pending_op(op_node)
            op_node.insert_input(var)
        for output in op_desc.output_arg_names():
            var = _create_node(name=output)
            var.set_gen_op(op_node)
            op_node.insert_output(var)
        return op_node

    # Record the forward vars
    forward_vars_set = set() if input_grad_names_set is None else set(
        input_grad_names_set)
    for op in forward_ops:
        forward_vars_set.update(op.desc.input_arg_names())
        forward_vars_set.update(op.desc.output_arg_names())

    # Record the vars which are created during backward and is not generated by op.
    backward_vars_set = set()
    # special_op_nodes is the candidate sub-graph head node.
    special_op_nodes = set()
    for op_desc in grad_op_descs:
        input_set = set(op_desc.input_arg_names())
        # The new_vars are created during backward and is not generated by op.
        new_vars = input_set - forward_vars_set - backward_vars_set
        backward_vars_set.update(op_desc.output_arg_names())

        op_node = _create_op_node(op_desc)
        if len(new_vars) == len(input_set):
            special_op_nodes.add(op_node)

    not_need_op_descs = []
    # Start traversing all candidate sub-graph headers to check whether
    # they are connected to backward computational graphs, and if they are
    # not, list them in not_need_op_descs
    for special_op_node in special_op_nodes:
        op_list = [special_op_node]
        ready_vars = set(special_op_node.inputs)
        remove_ops = True
        candidate_ops = [special_op_node]
        while len(candidate_ops) > 0:
            op_node = candidate_ops.pop(0)
            if _all_in_set_(op_node.inputs, ready_vars):
                for out_var in op_node.outputs:
                    candidate_ops.extend(out_var.pendding_ops)
                    op_list.extend(out_var.pendding_ops)
                ready_vars.update(op_node.outputs)
            else:
                remove_ops = False
                break
        if remove_ops:
            not_need_op_descs.extend([node.op_desc for node in op_list])

    return set(not_need_op_descs)


from .proto import framework_pb2


def serialize_op_decs(op_desc):
    protostr = op_desc.serialize_to_string()
    proto = framework_pb2.OpDesc.FromString(six.binary_type(protostr))
    return proto.__str__()


def _append_backward_ops_with_checkpoints_(
        block, ops, target_block, no_grad_dict, grad_to_var, checkpoints):
    """
    Create grad ops with forward ops, and insert them into given block

    Args:
        block(Block): the block where forward ops are
        ops(Op): the forward operators whose forward recomputation backward ops need to be added
        target_block(Block): the block which is going to hold new generated grad ops
        no_grad_dict(dict):
            key(int) block index
            val(str): corresponding forward variable name
        checkpoints: variables that a user defined as checkpoint for forward recomputation

    Algorithms:
        1) find ops between checkpoints, i.e. recompute_segments
        2) go through all forward ops and induct all variables that will be hold in memory
            a. variables that are used across segments will be held in memory
            b. output of dropout op will be held in memory
            c. input variables will be held in memory
        3) go through each recompute_segments, add backward ops with forward recomputation
            a. add ops in current recompute_segment as forward recomputation ops
            b. rename all non-checkpoint variables in recomputation ops
            c. add backward ops of current recomputation ops
            d. add sum op for repetitive_outputs
        4) remove no grad branch as it is in _remove_no_grad_branch_
        5) Note1: all appended ops' OpRole are Backward
        6) Note2: all variables with new name should be returned so that _append_backward_vars_ can be called
        7) Note3: current forward recomputation backpropagation does not handle programs with subblock
    """

    checkpoints_name = [x.name for x in checkpoints]
    checkpoints_name = list(set(checkpoints_name))
    local_block = block.program._create_block()
    buffer_block = block.program._create_block()

    # 1) find ops between checkpoints, i.e. recompute_segments
    program_stat = ProgramStats(block, ops)
    program_stat.build_stats()
    checkpoints_name = program_stat.sort_checkpoints(checkpoints_name)
    segments = []

    if len(checkpoints_name) == 1:
        # only one checkpoint
        max_op_idx = -1
        var_group = [checkpoints_name[0]]
        for name in var_group:
            if name not in program_stat.var_op_deps:
                break
            op_idx = program_stat.var_op_deps[name]["var_as_output_ops"]
            for idx in op_idx:
                max_op_idx = max(max_op_idx, idx)
        if max_op_idx > 0:
            segments.append([0, max_op_idx + 1])
    else:
        start_idx = 0
        while True:
            if start_idx >= len(checkpoints_name) - 1:
                break
            flag, min_idx, max_idx = program_stat.is_subgraph(
                [checkpoints_name[start_idx]],
                [checkpoints_name[start_idx + 1]])
            if flag:
                segments.append([min_idx, max_idx + 1])
            start_idx += 1

    if segments != [] and segments[0][0] != 0:
        recompute_segments = [[0, segments[0][0]]] + segments
    else:
        recompute_segments = segments

    # 2) go through all forward ops and induct all variables that will be hold in memory
    vars_should_be_hold = []
    # a. variables that are used across segments will be held in memory
    for segment in recompute_segments:
        vars_should_be_hold.extend(
            program_stat.get_out_of_subgraph_vars(segment[0], segment[1]))
    # b. output of dropout op will be held in memory
    vars_should_be_hold.extend(program_stat.get_reserved_vars())
    # c. input variables are checkpoints
    vars_should_be_hold.extend(program_stat.get_input_nodes())
    vars_should_be_hold = list(set(vars_should_be_hold))

    # 3) go through each recompute_segments, add backward ops with forward recomputation
    grad_op_descs = []
    var_name_dict = {}

    vars_in_memory = vars_should_be_hold + checkpoints_name

    max_calculated_op_position = len(ops)
    if recompute_segments == []:
        # if there is no recompute segment, add backward ops like
        # _append_backward_ops_ function
        gap_ops = ops[0:max_calculated_op_position]
        for op in reversed(gap_ops):
            if op.has_attr("sub_block"):
                raise Exception("Recompute don't support ops with sub_block"
                                "invoke op: %s" %
                                _pretty_op_desc_(op.desc, "with_sub_block"))
            grad_op_desc, op_grad_to_var = core.get_grad_op_desc(
                op.desc, cpt.to_text(no_grad_dict[block.idx]), [])
            added_descs = _add_descs_to_block(grad_op_desc, local_block)
            grad_op_descs.extend(added_descs)
            grad_to_var.update(op_grad_to_var)

    for i, segment in enumerate(recompute_segments[::-1]):
        # add grad op for ops not in any segments
        gap_ops = ops[segment[1]:max_calculated_op_position]
        max_calculated_op_position = segment[0]
        for op in reversed(gap_ops):
            if op.has_attr("sub_block"):
                raise Exception("Recompute don't support ops with sub_block"
                                "invoke op: %s" %
                                _pretty_op_desc_(op.desc, "with_sub_block"))
            grad_op_desc, op_grad_to_var = core.get_grad_op_desc(
                op.desc, cpt.to_text(no_grad_dict[block.idx]), [])
            added_descs = _add_descs_to_block(grad_op_desc, local_block)
            grad_op_descs.extend(added_descs)
            grad_to_var.update(op_grad_to_var)

        ff_ops = ops[segment[0]:segment[1]]
        var_suffix = ".subprog_%d" % i

        for op in ff_ops:
            if op.has_attr("sub_block"):
                raise Exception("Recompute don't support ops with sub_block"
                                "invoke op: %s" %
                                _pretty_op_desc_(op.desc, "with_sub_block"))
            input_and_output_names = []
            input_and_output_names.extend(op.desc.input_arg_names())
            input_and_output_names.extend(op.desc.output_arg_names())
            for name in input_and_output_names:
                if block.var(name).persistable or name in checkpoints_name:
                    continue
                if name in vars_should_be_hold:
                    continue
                if name not in var_name_dict:
                    var_name_dict[name] = name + var_suffix
        # 3.a. add ops in current recompute_segment as forward recomputation ops
        buffer_descs = _add_needed_descs_to_block(ff_ops, buffer_block, block,
                                                  vars_in_memory)
        added_descs = _add_descs_to_block(ff_ops, local_block)

        # 3.b. rename all non-checkpoint variables in recomputation ops
        for key in var_name_dict:
            _rename_arg_(buffer_descs, key, var_name_dict[key])

        # added_descs should be in grad_op_descs because it is backward op desc
        grad_op_descs.extend(buffer_descs)

        # 3.c. add backward ops of current recomputation ops
        for op_desc in reversed(added_descs):
            grad_op_desc, op_grad_to_var = core.get_grad_op_desc(
                op_desc, cpt.to_text(no_grad_dict[block.idx]), [])
            for key in var_name_dict:
                _rename_arg_(grad_op_desc, key, var_name_dict[key])
            grad_op_descs.extend(grad_op_desc)
            grad_to_var.update(op_grad_to_var)

    # 3.d. add sum op for repetitive_outputs
    grad_op_descs = _addup_repetitive_outputs_(grad_op_descs)
    # 4) remove no grad branch as it is in _remove_no_grad_branch_
    grad_op_descs = _remove_no_grad_branch_(grad_op_descs,
                                            no_grad_dict[block.idx])
    added_descs = _add_descs_to_block(grad_op_descs, target_block)
    return program_stat, checkpoints_name, vars_should_be_hold, recompute_segments


def _append_backward_ops_(block,
                          ops,
                          target_block,
                          no_grad_dict,
                          grad_to_var,
                          callbacks=None,
                          input_grad_names_set=None):
    """
    Create all grad ops, and insert them into given block

    Args:
        block(Block): the block where forward ops are
        ops(Op): the forward operators whose backward ops need to be added
        target_block(Block): the block which is going to hold new generated grad ops
        no_grad_dict(dict):
            key(int)  block index
            val(set) a set of varibale names. These varibales have no gradient
        grad_to_var(dict)(output argument):
            key(str): grad variable name
            val(str): corresponding forward variable name
        callbacks(callable object): a callable object used to decorate new generated grad ops
        input_grad_names_set(set): this set is used to store the gradients' name which is
            generated by backward ops, and input_grad_names_set can help to prune the unnecessary
            backward ops.
    """
    if callbacks is not None:
        assert (isinstance(callbacks, list))
        for cb in callbacks:
            if not hasattr(cb, '__call__'):
                raise ValueError("'callback' must be a callable object.")

    # grad_op_descs holds created grad_op, and will be appended to target_block
    grad_op_descs = []
    program = block.program
    for op in reversed(ops):
        grad_sub_block_list = []
        # If the op has its own sub-block, deal with the sub-block first
        if op.has_attr("sub_block"):
            sub_block = program.block(op._block_attr_id("sub_block"))
            grad_sub_block = program._create_block()
            grad_sub_block._set_forward_block_idx(sub_block.idx)
            # see follwing comments for why set None here.
            pre_input_grad_names_set = copy.copy(input_grad_names_set)
            input_grad_names_set = None
            _append_backward_ops_(sub_block, sub_block.ops, grad_sub_block,
                                  no_grad_dict, grad_to_var, callbacks,
                                  input_grad_names_set)
            input_grad_names_set = pre_input_grad_names_set

            program._rollback()
            grad_sub_block_list.append(grad_sub_block.desc)

        # Getting op's corresponding grad_op
        grad_op_desc, op_grad_to_var = core.get_grad_op_desc(
            op.desc, cpt.to_text(no_grad_dict[block.idx]), grad_sub_block_list)

        # If input_grad_names_set is not None, extend grad_op_descs only when
        # any input grad in outputs of previous grad ops.
        # But this strategy is not suited for while op for some control flow,
        # for example, for while op, the grads maybe generated in next loop.
        if input_grad_names_set is not None:
            is_append_grad = False
            for op_desc in grad_op_desc:
                input_grad_names = [
                    name for name in op_desc.input_arg_names()
                    if name.find(core.grad_var_suffix()) != -1
                ]
                # some code of gradient ops, like increment, are not very
                # standard, there is no @GRAD in these ops' inputs.
                if len(input_grad_names) == 0:
                    is_append_grad = True
                    break

                if _some_in_set_(input_grad_names, input_grad_names_set):
                    grad_op_descs.append(op_desc)
                    is_append_grad = True
                    for name in op_desc.output_arg_names():
                        input_grad_names_set.add(name)
            if is_append_grad:
                grad_to_var.update(op_grad_to_var)
        else:
            grad_op_descs.extend(grad_op_desc)
            grad_to_var.update(op_grad_to_var)

    # add grad_op_desc by reversed ops

    # sum parameter's gradients' var given multiple var gradient
    grad_op_descs = _addup_repetitive_outputs_(grad_op_descs)

    # if all outputs of the grad op are in no_grad_set, then just remove and fill zero
    # if all inputs of the grad op are in no_grad_set, just remove this op
    grad_op_descs = _remove_no_grad_branch_(grad_op_descs,
                                            no_grad_dict[block.idx])

    # remove some backward ops
    not_need_ops = _find_not_need_ops(grad_op_descs, ops, input_grad_names_set)

    grad_op_descs = [
        op_desc for op_desc in grad_op_descs if op_desc not in not_need_ops
    ]
    # append op_desc in grad_op_descs to target_block
    op_role_attr_name = core.op_proto_and_checker_maker.kOpRoleAttrName()
    backward = core.op_proto_and_checker_maker.OpRole.Backward
    for op_desc in grad_op_descs:
        new_op_desc = target_block.desc.append_op()
        new_op_desc.copy_from(op_desc)
        new_op_desc._set_attr(op_role_attr_name, backward)
        grad_to_var["__current_op_desc__"] = new_op_desc
        if callbacks is not None:
            assert (isinstance(callbacks, list))
            for cb in callbacks:
                cb(block=target_block, context=grad_to_var)


def _append_backward_vars_(block, start_op_idx, grad_to_var, grad_info_map):
    """
    Create new variables required by backward pass.

    Args:
        block(Block): the block where new variables will be created
        start_op_idx(int): Only variables required by ops in block.ops[start_op_idx : ] will be created
        grad_to_var(dict):
            key(str): grad variable name
            val(str): corresponding forward variable name
            In most cases, this dict is generated by _append_backward_ops_()
        grad_info_map(dict)(output argument):
            key(str): forward variable name
            val(tuple): a tuple of (str, Block), str is the corresponding grad name, Block is the block containing grad variable
    """
    for op_idx in range(start_op_idx, block.desc.op_size()):
        op_desc = block.desc.op(op_idx)
        if op_desc.has_attr("sub_block"):
            sub_block = block.program.block(op_desc._block_attr_id("sub_block"))
            _append_backward_vars_(sub_block, 0, grad_to_var, grad_info_map)
        new_vars = set()
        # create new gradient variables
        for grad_var_name in op_desc.output_arg_names():
            if block.desc.has_var_recursive(cpt.to_bytes(
                    grad_var_name)) or grad_var_name == core.empty_var_name():
                continue
            block.desc.var(cpt.to_bytes(grad_var_name))
            new_vars.add(grad_var_name)
            if grad_var_name not in grad_to_var:
                continue
            grad_info_map[grad_to_var[grad_var_name]] = (grad_var_name, block)
        # infer_shape and infer_type
        op_desc.infer_var_type(block.desc)
        op_desc.infer_shape(block.desc)
        for arg in op_desc.output_arg_names():
            if arg in new_vars:
                _infer_var_data_type_(arg, block)


def _rename_grad_(block, start_op_idx, grad_to_var, target_grad_map):
    var_map = copy.copy(target_grad_map)
    for op_idx in range(start_op_idx, block.desc.op_size()):
        op_desc = block.desc.op(op_idx)
        for name in op_desc.input_arg_names():
            if name in var_map:
                op_desc._rename_input(name, var_map[name])

        for name in op_desc.output_arg_names():
            if "@GRAD" not in name:
                continue
            if block.desc.find_var(name.encode("ascii")):
                new_name = unique_name.generate(name)
                op_desc._rename_output(name, new_name)
                var_map[name] = new_name

    for g, ng in six.iteritems(var_map):
        if g in grad_to_var:
            grad_to_var[ng] = grad_to_var[g]
            grad_to_var.pop(g)


def _get_stop_gradients_(program):
    no_grad_dict = dict()
    assert isinstance(program, framework.Program)
    for block in program.blocks:
        assert isinstance(block, framework.Block)
        block_no_grad_set = set()
        for var in list(block.vars.values()):
            assert isinstance(var, framework.Variable)
            if var.stop_gradient:
                block_no_grad_set.add(_append_grad_suffix_(var.name))
        no_grad_dict[block.idx] = block_no_grad_set
    return no_grad_dict


def append_backward(loss,
                    parameter_list=None,
                    no_grad_set=None,
                    callbacks=None,
                    checkpoints=None):
    """
    This function appends backward part to main_program.

    A complete neural network training is made up of forward and backward
    propagation. However, when we configure a network, we only need to
    specify its forward part. This function uses the chain rule to automatically
    generate the backward part according to the forward part.

    In most cases, users do not need to invoke this function manually.
    It will be automatically invoked by the optimizer's `minimize` function.

    Parameters:
        loss( :ref:`api_guide_Variable_en` ): The loss variable of the network.
        parameter_list(list of str, optional): Names of parameters that need
                                           to be updated by optimizers.
                                           If it is None, all parameters
                                           will be updated.
                                           Default: None.
        no_grad_set(set of str, optional): Variable names in the :ref:`api_guide_Block_en` 0 whose gradients
                               should be ignored. All variables with
                               `stop_gradient=True` from all blocks will
                               be automatically added into this set.
                               If this parameter is not None, the names in this set will be added to the default set.
                               Default: None.
        callbacks(list of callable object, optional): List of callback functions.
                                               The callbacks are used for
                                               doing some custom jobs during
                                               backward part building. All
                                               callable objects in it will
                                               be invoked once each time a
                                               new gradient operator is added
                                               into the program. The callable
                                               object must has two input
                                               parameters: 'block' and 'context'.
                                               The 'block' is the :ref:`api_guide_Block_en` which
                                               the new gradient operator will
                                               be added to. The 'context' is a
                                               map, whose keys are gradient
                                               variable names and values are
                                               corresponding original :ref:`api_guide_Variable_en` .
                                               In addition to this, the 'context'
                                               has another special key-value pair:
                                               the key is string '__current_op_desc__'
                                               and the value is the op_desc of the
                                               gradient operator who has just
                                               triggered the callable object.
                                               Default: None.

    Returns:
        list of tuple ( :ref:`api_guide_Variable_en` , :ref:`api_guide_Variable_en` ): Pairs of parameter and its corresponding gradients.
        The key is the parameter and the value is gradient variable.

    Raises:
        AssertionError: If `loss` is not an instance of Variable.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.data(name='x', shape=[None, 13], dtype='float32')
            y = fluid.data(name='y', shape=[None, 1], dtype='float32')

            y_predict = fluid.layers.fc(input=x, size=1, act=None)
            loss = fluid.layers.square_error_cost(input=y_predict, label=y)

            avg_loss = fluid.layers.mean(loss)
            param_grad_list = fluid.backward.append_backward(loss=avg_loss)
            p_g_list1 = fluid.backward.append_backward(loss=avg_loss)  # len(p_g_list1) == 2
            p_g_list2 = fluid.backward.append_backward(loss=avg_loss, parameter_list=[p_g_list1[0][0].name])  # len(p_g_list1) == 1
            p_g_list3 = fluid.backward.append_backward(loss=avg_loss, no_grad_set=set([p_g_list1[0][0].name]))  # len(p_g_list1) == 1
            p_g_list4 = fluid.backward.append_backward(loss=avg_loss, parameter_list=[p_g_list1[0][0].name], no_grad_set=set([p_g_list1[0][0].name]))  # len(p_g_list1) == 0

    """
    assert isinstance(loss, framework.Variable)

    if loss.op is None:
        # the loss is from a cloned program. Find loss op manually.
        _find_loss_op_(loss)

    loss.op._set_attr(core.op_proto_and_checker_maker.kOpRoleAttrName(),
                      int(core.op_proto_and_checker_maker.OpRole.Forward) |
                      int(core.op_proto_and_checker_maker.OpRole.Loss))

    if callbacks is not None:
        isinstance(callbacks, list)

    program = loss.block.program
    program._appending_grad_times += 1

    if no_grad_set is None:
        no_grad_set = set()
    no_grad_set = copy.copy(no_grad_set)
    no_grad_dict = _get_stop_gradients_(program)
    no_grad_dict[0].update(list(map(_append_grad_suffix_, no_grad_set)))

    grad_info_map = dict()
    root_block = program.block(0)

    fwd_op_num = root_block.desc.op_size()
    current_block_idx = program.current_block_idx
    grad_to_var = dict()

    op_desc = _create_loss_op_desc_(loss)
    root_block.desc.append_op().copy_from(op_desc)

    block_no_grad_set = set(map(_strip_grad_suffix_, no_grad_dict[0]))
    op_path = _find_op_path_(root_block, [loss], [], block_no_grad_set)
    no_grad_vars = _find_no_grad_vars(root_block, op_path, [loss],
                                      block_no_grad_set)
    block_no_grad_set.update(no_grad_vars)
    no_grad_dict[0].update(list(map(_append_grad_suffix_, block_no_grad_set)))

    input_grad_names_set = None
    # For double backward, input_grad_names is used for filter
    # some non-used gradients op.
    if program._appending_grad_times > 1:
        input_grad_names_set = set([_append_grad_suffix_(loss.name)])


    if checkpoints != None and \
       isinstance(checkpoints, list) and \
       len(checkpoints) > 0:
        program_stat, checkpoint_names, \
        vars_should_be_hold, \
        recompute_segments = \
                        _append_backward_ops_with_checkpoints_(
                            root_block,
                            op_path,
                            root_block,
                            no_grad_dict,
                            grad_to_var,
                            checkpoints)
    else:
        _append_backward_ops_(
            root_block,
            op_path,
            root_block,
            no_grad_dict,
            grad_to_var,
            callbacks,
            input_grad_names_set=input_grad_names_set)

    # Because calc_gradient may be called multiple times,
    # we need rename the internal gradient variables so that they have
    # different names.
    _rename_grad_(root_block, fwd_op_num, grad_to_var, {})

    _append_backward_vars_(root_block, fwd_op_num, grad_to_var, grad_info_map)

    program.current_block_idx = current_block_idx
    program._sync_with_cpp()

    if parameter_list is not None:
        parameters = parameter_list
    else:
        params = program.global_block().all_parameters()
        parameters = [param.name for param in params if param.trainable]

    params_and_grads = []
    for param in parameters:
        if cpt.to_text(param) not in grad_info_map:
            continue
        grad_info = grad_info_map[param]
        grad_block = grad_info[1]
        if not grad_block.has_var(grad_info[0]):
            raise ValueError("grad block[{0}] did not have grad var {1}".format(
                grad_info[1], grad_info[0]))
        # Get the param var from the global block
        param_var = program.global_block().var(param)
        grad_var = grad_block.var(grad_info[0])
        if loss.block.has_var(grad_info[0]):
            params_and_grads.append((param_var, grad_var))
        else:
            params_and_grads.append((param_var, None))

    op_role_var_attr_name = core.op_proto_and_checker_maker.kOpRoleVarAttrName()
    for p, g in params_and_grads:
        if g is None:
            continue
        for op in reversed(program.global_block().ops):
            assert isinstance(op, framework.Operator)
            if g.name in op.output_arg_names:
                g.op = op
                break

        if g.op is None:
            raise ValueError("Unexpected branch")
        attr_val = [p.name, g.name]
        if g.op.has_attr(op_role_var_attr_name):
            attr_val.extend(g.op.attr(op_role_var_attr_name))
        g.op._set_attr(op_role_var_attr_name, attr_val)

    return params_and_grads


def _as_list(x):
    if x is None:
        return []
    return list(x) if isinstance(x, collections.Sequence) else [x]


def _find_no_grad_vars(block, op_path, targets, no_grad_set):
    """
    Find the vars which is not used in the program, and
    those var belong to no_grad_var.
    """
    output_names = set([out.name for out in targets])
    no_grad_var = []
    for i, op in reversed(list(enumerate(op_path))):
        # If the op has sub_block, it is too complicated to find the correct no_grad_var.
        if not op.has_attr("sub_block"):
            for out_var in op.desc.output_arg_names():
                if out_var not in output_names and out_var not in op.desc.input_arg_names(
                ) and not block.vars[out_var].stop_gradient:
                    no_grad_var.append(out_var)
        for name in op.desc.input_arg_names():
            if name not in no_grad_set:
                output_names.add(name)
    return set(no_grad_var)


def _find_op_path_(block, outputs, inputs, no_grad_set):
    """
    no_grad_set will also be changed
    """
    input_names = set([inp.name for inp in inputs])
    output_names = set([out.name for out in outputs])

    relevant_op_flags = [True] * len(block.ops)

    # All the inputs of the block are used if inputs is empty,
    if inputs:
        for i, op in enumerate(block.ops):
            if _some_in_set_(op.desc.input_arg_names(), input_names):
                for name in op.desc.output_arg_names():
                    if name not in no_grad_set:
                        input_names.add(name)
            else:
                relevant_op_flags[i] = False

    for i, op in reversed(list(enumerate(block.ops))):
        if _some_in_set_(op.desc.output_arg_names(), output_names):
            for name in op.desc.input_arg_names():
                if name not in no_grad_set:
                    output_names.add(name)
        else:
            relevant_op_flags[i] = False

    op_path = [
        block.ops[i] for i in range(len(block.ops)) if relevant_op_flags[i]
    ]

    if inputs:
        for op in op_path:
            for name in op.desc.input_arg_names():
                if name not in input_names and block.vars[name].stop_gradient:
                    no_grad_set.add(name)

    return op_path


def calc_gradient(targets, inputs, target_gradients=None, no_grad_set=None):
    """
    Backpropagate the gradients of targets to inputs.

    Args:
        targets(Variable|list[Variable]): The target variables
        inputs(Variable|list[Variable]): The input variables
        target_gradients (Variable|list[Variable]|None): The gradient variables
            of targets which has the same shape with targets, If None, ones will
            be created for them.
        no_grad_set(set[string]): The names of variables that have no gradients
            in Block 0. All variables with `stop_gradient=True` from all blocks
            will be automatically added.

    Return:
        (list[Variable]): A list of gradients for inputs
        If an input does not affect targets, the corresponding gradient variable
        will be None
    """
    targets = _as_list(targets)
    inputs = _as_list(inputs)
    target_gradients = _as_list(target_gradients)

    block = targets[0].block
    prog = block.program
    # increase appending gradients times
    prog._appending_grad_times += 1
    block_idx = block.idx

    if not target_gradients:
        target_gradients = [None] * len(targets)

    if len(targets) != len(target_gradients):
        raise ValueError(
            "Should have the same number of target_gradients as targets")

    if no_grad_set is None:
        no_grad_set = set()
    no_grad_set = copy.copy(no_grad_set)
    no_grad_dict = _get_stop_gradients_(prog)
    no_grad_dict[0].update(list(map(_append_grad_suffix_, no_grad_set)))

    fwd_op_num = block.desc.op_size()

    input_grad_names_set = set()

    target_grad_map = {}
    for i, grad in enumerate(target_gradients):
        target = targets[i]
        if grad is None:
            grad_name = _append_grad_suffix_(target.name)
            target_shape = paddle.fluid.layers.shape(target)
            op_desc = _create_op_desc_("fill_constant",
                                       {"ShapeTensor": [target_shape.name]},
                                       {"Out": [grad_name]}, {
                                           "shape": target.shape,
                                           "value": 1.0,
                                           "dtype": target.dtype,
                                       })

            block.desc.append_op().copy_from(op_desc)
            input_grad_names_set.add(grad_name)
        else:
            if target.block.idx != block_idx or target.block.program != prog:
                raise ValueError("all targets must be in the same block")
            if target.shape != grad.shape:
                raise ValueError(
                    "The shapes of target and grad are different: %s %s" % (
                        target.name, grad.name))
            target_grad_map[_append_grad_suffix_(target.name)] = grad.name
            input_grad_names_set.add(grad.name)

    # For double backward, input_grad_names is used for filter
    # some non-used gradients op.
    if prog._appending_grad_times == 1:
        input_grad_names_set = None

    for input in inputs:
        if input.block.program != prog:
            raise "input must be in the same program as targets"

    block_no_grad_set = set(map(_strip_grad_suffix_, no_grad_dict[0]))
    op_path = _find_op_path_(block, targets, inputs, block_no_grad_set)
    no_grad_dict[0].update(list(map(_append_grad_suffix_, block_no_grad_set)))
    grad_to_var = dict()
    grad_info_map = dict()
    _append_backward_ops_(
        block,
        op_path,
        block,
        no_grad_dict,
        grad_to_var,
        input_grad_names_set=input_grad_names_set)

    # Because calc_gradient may be called multiple times,
    # we need rename the internal gradient variables so that they have
    # different names.
    _rename_grad_(block, fwd_op_num, grad_to_var, target_grad_map)

    _append_backward_vars_(block, fwd_op_num, grad_to_var, grad_info_map)
    prog._sync_with_cpp()

    grad_vars = []
    for input_var in inputs:
        if input_var.name not in grad_info_map:
            grad_vars.append(None)
        else:
            grad_info = grad_info_map[input_var.name]
            grad_block = grad_info[1]
            grad_var = grad_block.var(grad_info[0])
            grad_vars.append(grad_var)

    if len(grad_vars) == 1:
        return grad_vars[0]
    else:
        return grad_vars


def gradients(targets, inputs, target_gradients=None, no_grad_set=None):
    """
    Backpropagate the gradients of targets to inputs.

    Args:
        targets (Variable|list[Variable]): The target variables.
        inputs (Variable|list[Variable]): The input variables.
        target_gradients (Variable|list[Variable]|None): The gradient variables
            of targets which has the same shape with targets, If None, ones will
            be created for them.
        no_grad_set (set[string]): The names of variables that have no gradients
            in Block 0. All variables with `stop_gradient=True` from all blocks
            will be automatically added.

    Return:
        (list[Variable]): A list of gradients for inputs
        If an input does not affect targets, the corresponding gradient variable
        will be None.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            x = fluid.layers.data(name='x', shape=[2,8,8], dtype='float32')
            x.stop_gradient=False
            y = fluid.layers.conv2d(x, 4, 1, bias_attr=False)
            y = fluid.layers.relu(y)
            y = fluid.layers.conv2d(y, 4, 1, bias_attr=False)
            y = fluid.layers.relu(y)
            z = fluid.gradients([y], x)
            print(z)
    """
    outs = calc_gradient(targets, inputs, target_gradients, no_grad_set)
    return _as_list(outs)
