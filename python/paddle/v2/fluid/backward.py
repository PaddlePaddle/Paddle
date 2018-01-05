from paddle.v2.fluid import framework as framework
from . import core
import collections

__all__ = ['append_backward']


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
        op_desc.rename_input(old_name, new_name)
        op_desc.rename_output(old_name, new_name)


def _create_op_desc_(op_type, inputs, outputs, attrs):
    """
    Create a C++ OpDesc object with specified inputs, outputs and attributes.
    """
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


def _infer_var_data_type_(grad_var_name, block):
    """
    Infer the data type of given grad variable
    """
    grad_var = block.desc.find_var(grad_var_name.encode("ascii"))
    fwd_name = _strip_grad_suffix_(grad_var_name.encode("ascii"))
    if block.desc.has_var_recursive(fwd_name):
        fwd_var = block.desc.find_var_recursive(fwd_name.encode("ascii"))
        grad_var.set_dtype(fwd_var.dtype())
    else:
        grad_var.set_dtype(core.DataType.FP32)


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


def _strip_grad_suffix_(name):
    """
    Strip the grad suffix from the given varibale name
    e.g. x@GRAD ==> x
         y@GRAD@RENAME@1 ==> y
    """
    pos = name.find(core.grad_var_suffix())
    return name[:pos] if pos != -1 else name


def _append_grad_suffix_(name):
    """
    Append grad suffix to the given variable name
    e.g. x ==> x@GRAD
    """
    return name + core.grad_var_suffix()


def _addup_repetitive_outputs_(op_descs):
    """
    In backward part, an variable may be the output of more than one ops.
    In this case, the variable should be the accumulation of all the outputs.
    `sum_op`s are added to implement the accumulate.
    """
    pending_sum_ops = []
    var_rename_count = collections.defaultdict(int)
    renamed_vars = collections.defaultdict(list)
    for idx, op_desc in enumerate(op_descs):
        for var_name in op_desc.input_arg_names():
            if len(renamed_vars[var_name]) > 1:
                pending_sum_ops.append(
                    (_create_op_desc_("sum", {"X": renamed_vars[var_name]},
                                      {"Out": [var_name]}, {}), idx))
                renamed_vars[var_name] = [var_name]
        for var_name in op_desc.output_arg_names():
            if var_name == core.empty_var_name(
            ) or var_name in op_desc.input_arg_names():
                # empty variable or inplace op
                continue
            if len(renamed_vars[var_name]) == 0:
                # it's the first time we get the variable
                renamed_vars[var_name] = [var_name]
            else:
                if len(renamed_vars[var_name]) == 1:
                    new_name = var_name + "@RENAME@" + \
                        str(var_rename_count[var_name])
                    var_rename_count[var_name] += 1
                    # rename original var_name
                    renamed_vars[var_name][0] = new_name
                    _rename_arg_(op_descs, var_name, new_name, 0, idx)
                    _rename_arg_(pending_sum_ops, var_name, new_name)

                new_name = var_name + "@RENAME@" + \
                    str(var_rename_count[var_name])
                var_rename_count[var_name] += 1
                op_desc.rename_output(var_name, new_name)
                renamed_vars[var_name].append(new_name)
    for var_name, inputs in renamed_vars.iteritems():
        if len(inputs) > 1:
            pending_sum_ops.append((_create_op_desc_(
                "sum", {"X": inputs}, {"Out": [var_name]}, {}), len(op_descs)))
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
        if _all_in_set_(
                filter(lambda name: name.find(core.grad_var_suffix()) != -1,
                       op_desc.input_arg_names()), no_grad_set):
            no_grad_set.union(out_arg_names)
            return True
        return False

    # Remove ops whose outputs are all in no_grad_dict
    op_descs = filter(
        lambda op_desc: not _op_can_be_removed_(op_desc, no_grad_set), op_descs)
    # Insert fill_zeros_like_op
    to_insert = []
    for idx, op_desc in enumerate(op_descs):
        for arg in op_desc.input_arg_names():
            if core.grad_var_suffix() in arg and arg in no_grad_set:
                to_insert.append((_create_op_desc_("fill_zeros_like", {
                    "X": [_strip_grad_suffix_(arg)]
                }, {"Y": [arg]}, {}), idx))

    map(lambda p: op_descs.insert(p[1], p[0]), reversed(to_insert))

    return op_descs


def _append_backward_ops_(target,
                          block,
                          target_block,
                          no_grad_dict,
                          grad_to_var,
                          callback=None):
    """
    Create all grad ops, and insert them into given block

    Args:
        target(Variable): the target variable of forward pass
        block(Block): the block where forward ops are
        target_block(Block): the block which is going to hold new generated grad ops
        no_grad_dict(dict): 
            key(int)  block index
            val(set) a set of varibale names. These varibales have no gradient
        grad_to_var(dict)(output argument):
            key(str): grad variable name
            val(str): corresponding forward variable name
        callback(callable object): a callable object used to decorate new generated grad ops
    """
    if callback is None:

        def empty_callback(block):
            pass

        callback = empty_callback
    elif not hasattr(callback, '__call__'):
        raise ValueError("'callback' must be a callable object.")

    # grad_op_descs holds created grad_op, and will be appended to target_block
    grad_op_descs = []
    program = block.program
    for op in reversed(block.ops):
        grad_sub_block_list = []
        # If the op has its own sub-block, deal with the sub-block first
        if op.has_attr("sub_block"):
            sub_block = program.block(op.block_attr("sub_block"))
            grad_sub_block = program.create_block(parent_idx=sub_block.idx)
            _append_backward_ops_(target, sub_block, grad_sub_block,
                                  no_grad_dict, grad_to_var, callback)
            grad_sub_block_list.append(grad_sub_block.desc)

        # Getting op's corresponding grad_op
        grad_op_desc, op_grad_to_var = core.get_grad_op_desc(
            op.desc, no_grad_dict[block.idx], grad_sub_block_list)
        grad_op_descs.extend(grad_op_desc)
        grad_to_var.update(op_grad_to_var)

    grad_op_descs = _addup_repetitive_outputs_(grad_op_descs)

    grad_op_descs = _remove_no_grad_branch_(grad_op_descs,
                                            no_grad_dict[block.idx])

    if target_block.idx == 0:
        grad_op_descs.insert(
            0,
            _create_op_desc_("fill_constant", {}, {
                "Out": [_append_grad_suffix_(target.name)]
            }, {"shape": [1],
                "value": 1.0,
                "dtype": target.dtype}))
    # append op_desc in grad_op_descs to target_block
    for op_desc in grad_op_descs:
        new_op_desc = target_block.desc.append_op()
        new_op_desc.copy_from(op_desc)
        callback(block=target_block, context=grad_to_var)


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
            val(tuple): a tuple of (str, int), str is the corresponding grad name, int is the block index
    """
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


def append_backward(loss, parameter_list=None, no_grad_set=None, callback=None):
    """
    Append backward part to main_program

    Args:
        loss(Variable): The variable generated by cost function.
        parameter_list(list): Parameters that need to be updated by optimizer.
            If None, it means all parameters need to be updated.
        no_grad_set(set): Variables that have no gradients in Block 0. 
            If None, the set will be generated inside the function and 
            contains all variables with `step_gradient=True` from all blocks.

    Return:
        (list[Variable]): list of (parameters, gradients) pair.
    """
    assert isinstance(loss, framework.Variable)

    program = loss.block.program
    no_grad_dict = dict()
    if no_grad_set is None:
        assert isinstance(program, framework.Program)
        for block in program.blocks:
            assert isinstance(block, framework.Block)
            block_no_grad_set = set()
            for var in block.vars.itervalues():
                assert isinstance(var, framework.Variable)
                if var.stop_gradient:
                    block_no_grad_set.add(_append_grad_suffix_(var.name))
            no_grad_dict[block.idx] = block_no_grad_set
    elif isinstance(no_grad_set, set):
        no_grad_dict = {
            0: set([_append_grad_suffix_(name) for name in no_grad_set])
        }
    else:
        raise ValueError("'no_grad_set' should be a set or None.")

    grad_info_map = dict()
    root_block = program.block(0)

    fwd_op_num = root_block.desc.op_size()
    current_block_idx = program.current_block_idx
    grad_to_var = dict()

    _append_backward_ops_(loss, root_block, root_block, no_grad_dict,
                          grad_to_var, callback)
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
