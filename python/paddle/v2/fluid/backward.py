from paddle.v2.fluid import framework as framework
from . import core
import collections
import pdb

__all__ = ['append_backward']


def _rename_arg_(op_desc_list, old_name, new_name, begin_idx=None,
                 end_idx=None):
    if begin_idx is None:
        begin_idx = 0
    if end_idx is None:
        end_idx = len(op_desc_list)
    for i in range(begin_idx, end_idx):
        op_desc = op_desc_list[i]
        if isinstance(op_desc, tuple):
            op_desc = op_desc[0]
        op_desc.rename_input(old_name, new_name)
        op_desc.rename_output(old_name, new_name)


def _create_op_desc_(op_type, inputs, outputs, attrs):
    op_desc = core.OpDesc()
    op_desc.set_type(op_type)
    for para, args in inputs.iteritems():
        op_desc.set_input(para, args)
    for para, args in outputs.iteritems():
        op_desc.set_output(para, args)
    for name, val in attrs.iteritems():
        if isinstance(val, framework.Block):
            op_desc.set_block_attr(name, val.desc)
        else:
            op_desc.set_attr(name, val)
    return op_desc


def _infer_var_data_type_(var_name, block):
    grad_var = block.desc.find_var(var_name.encode("ascii"))
    fwd_name = _strip_grad_suffix_(var_name.encode("ascii"))
    if block.desc.has_var_recursive(fwd_name):
        fwd_var = block.desc.find_var_recursive(fwd_name.encode("ascii"))
        grad_var.set_dtype(fwd_var.dtype())
    else:
        grad_var.set_dtype(core.DataType.FP32)


def _is_all_in_set_(cands, s):
    for c in cands:
        if not c in s:
            return False
    return True


def _strip_grad_suffix_(name):
    pos = name.find(core.grad_var_suffix())
    return name[:pos] if pos != -1 else name


def _append_grad_suffix_(name):
    return name + core.grad_var_suffix()


def _append_backward_ops_(target,
                          block,
                          target_block,
                          no_grad_set,
                          callback=None):
    grad_op_descs = []
    grad_to_var = dict()
    program = block.program
    for each_op in reversed(block.ops):
        grad_sub_block_list = []
        if each_op.has_attr("sub_block"):
            sub_block_idx = each_op.block_attr("sub_block")
            sub_block = program.block(sub_block_idx)
            grad_sub_block = program.create_block(parent_idx=sub_block_idx)
            sub_grad_to_var = _append_backward_ops_(
                target, sub_block, grad_sub_block, no_grad_set, callback)
            grad_to_var = dict(grad_to_var, **sub_grad_to_var)
            grad_sub_block_list.append(grad_sub_block.desc)
        grad_op_desc, op_grad_to_var = core.get_grad_op_desc(
            each_op.desc, no_grad_set[block.idx], grad_sub_block_list)
        grad_op_descs.append(grad_op_desc)
        grad_to_var = dict(grad_to_var, **op_grad_to_var)
    # grad_op_descs = [[op1_g1, op1_g2], [op2_g], ...]
    # flatten grad_op_descs
    grad_op_descs = [op for sublist in grad_op_descs for op in sublist]  # ?????

    pending_sum_ops = []
    var_rename_count = collections.defaultdict(int)
    var_inputs = collections.defaultdict(list)
    for idx, op_desc in enumerate(grad_op_descs):
        for var_name in op_desc.input_arg_names():
            if len(var_inputs[var_name]) > 1:
                pending_sum_ops.append((_create_op_desc_(
                    op_type="sum",
                    inputs={"X": var_inputs[var_name]},
                    outputs={"Out": [var_name]},
                    attrs={}), idx))
                var_inputs[var_name] = [var_name]
        for var_name in op_desc.output_arg_names():
            if var_name in op_desc.input_arg_names():
                # in place operator
                continue
            if var_name == core.empty_var_name() or len(var_inputs[
                    var_name]) == 0:
                # it's the first time we get the variable
                var_inputs[var_name] = [var_name]
            else:
                if len(var_inputs[var_name]) == 1:
                    new_name = var_name + "@RENAME@" + \
                        str(var_rename_count[var_name])
                    var_rename_count[var_name] = var_rename_count[var_name] + 1
                    # rename original var_name
                    var_inputs[var_name][0] = new_name
                    _rename_arg_(grad_op_descs, var_name, new_name, 0, idx)
                    _rename_arg_(pending_sum_ops, var_name, new_name)

                new_name = var_name + "@RENAME@" + \
                    str(var_rename_count[var_name])
                var_rename_count[var_name] = var_rename_count[var_name] + 1
                op_desc.rename_output(var_name, new_name)
                var_inputs[var_name].append(new_name)
    for var_name, inputs in var_inputs.iteritems():
        if len(inputs) > 1:
            pending_sum_ops.append((_create_op_desc_(
                op_type="sum",
                inputs={"X": inputs},
                outputs={"Out": [var_name]},
                attrs={}), len(grad_op_descs)))
    # sum_op descs are sorted according to their insert position
    for p in reversed(pending_sum_ops):
        grad_op_descs.insert(p[1], p[0])
    # Remove ops whose outputs are all in no_grad_set
    grad_op_descs = filter(
        lambda op_desc: not _is_all_in_set_(op_desc.output_arg_names(), no_grad_set[block.idx]),
        grad_op_descs)
    # Insert fill_zeros_like_op
    to_insert = []
    for idx, op_desc in enumerate(grad_op_descs):
        for arg in op_desc.input_arg_names():
            if core.grad_var_suffix() in arg and arg in no_grad_set[block.idx]:
                to_insert.append((arg, idx))
    for ele in reversed(to_insert):
        arg = ele[0]
        fill_zeros_like_op = _create_op_desc_(
            "fill_zeros_like", {"X": [_strip_grad_suffix_(arg)]}, {"Y": [arg]},
            {})
        grad_op_descs.insert(ele[1], fill_zeros_like_op)

    if target_block.idx == 0:
        grad_target_name = _append_grad_suffix_(target.name)
        # target_block.desc.var(grad_target_name.encode("ascii"))
        grad_op_descs.insert(
            0,
            _create_op_desc_(
                op_type="fill_constant",
                inputs={},
                outputs={"Out": [grad_target_name]},
                attrs={"shape": [1],
                       "value": 1.0,
                       "dtype": target.dtype}))
    for op_desc in grad_op_descs:
        new_op_desc = target_block.desc.append_op()
        new_op_desc.copy_from(op_desc)

    return grad_to_var


def _append_backward_vars_(block, start_op_idx, grad_to_var, grad_info_map):
    for op_idx in range(start_op_idx, block.desc.op_size()):
        op_desc = block.desc.op(op_idx)
        if op_desc.has_attr("sub_block"):
            sub_block = block.program.block(op_desc.block_attr("sub_block"))
            _append_backward_vars_(sub_block, 0, grad_to_var, grad_info_map)
        new_vars = set()
        # create new gradient variables
        for grad_var_name in op_desc.output_arg_names():
            grad_var_name = grad_var_name.encode("ascii")
            if block.desc.has_var_recursive(
                    grad_var_name) or grad_var_name == core.empty_var_name():
                continue
            block.desc.var(grad_var_name)
            new_vars.add(grad_var_name)
            if not grad_to_var.has_key(grad_var_name):
                continue
            grad_info_map[grad_to_var[grad_var_name]] = (grad_var_name, block)
        # infer_shape and infer_type
        op_desc.infer_var_type(block.desc)
        op_desc.infer_shape(block.desc)
        for arg in op_desc.output_arg_names():
            if arg in new_vars:
                _infer_var_data_type_(arg, block)


def append_backward(loss, parameter_list=None, no_grad_set=None):
    """
    Create and add gradient Operators in BlockDesc to compute
    gradients of `loss` for parameters in parameter_list

    :param loss: an variable generated by cost function.
    :type loss: Variable
    :param no_grad_set: variable that should not create gradient
    :type no_grad_set: set
    :param parameter_list: parameters that need to compute gradient and 
    update to optimize the lost.
    :type: list
    :return: list of (parameters, gradients) pair.
    :rtype: list[Variable]
    """
    assert isinstance(loss, framework.Variable)

    program = loss.block.program
    if no_grad_set is None:
        no_grad_set = dict()
        assert isinstance(program, framework.Program)
        for block in program.blocks:
            assert isinstance(block, framework.Block)
            block_no_grad_set = set()
            for var in block.vars.itervalues():
                assert isinstance(var, framework.Variable)
                if var.stop_gradient:
                    block_no_grad_set.add(_append_grad_suffix_(var.name))
            no_grad_set[block.idx] = block_no_grad_set
    else:
        # FIX ME
        no_grad_set = {0: no_grad_set}

    grad_info_map = dict()
    root_block = program.block(0)

    fwd_op_num = root_block.desc.op_size()
    current_block_idx = program.current_block_idx
    grad_to_var = _append_backward_ops_(loss, root_block, root_block,
                                        no_grad_set)
    _append_backward_vars_(root_block, fwd_op_num, grad_to_var, grad_info_map)
    program.current_block_idx = current_block_idx
    program.sync_with_cpp()

    if parameter_list is not None:
        parameters = parameter_list
    else:
        params = program.global_block().all_parameters()
        parameters = [param.name for param in params]
    params_and_grads = []
    for param in parameters:
        if param not in grad_info_map:
            raise ValueError("param %s is not in map" % param)
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
    return params_and_grads
