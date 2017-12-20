from paddle.v2.fluid import framework as framework
from . import core
import collections
import pdb

__all__ = ['append_backward_ops']


def rename_arg(op_desc_list, old_name, new_name, begin_idx=None, end_idx=None):
    if begin_idx is None:
        begin_idx = 0
    if end_idx is None:
        end_idx = len(op_desc_list)
    for i in range(begin_idx, end_idx):
        op_desc_list[i].rename_input(old_name, new_name)
        op_desc_list[i].rename_output(old_name, new_name)


def backward_impl(target,
                  block,
                  target_block,
                  no_grad_set,
                  grad_info_map,
                  callback=None):
    grad_op_descs = []
    grad_to_var = {}
    program = block.program
    for each_op in block.ops:
        grad_sub_block_list = []
        if each_op.has_attr("sub_block"):
            sub_block_idx = each_op.block_attr("sub_block")
            sub_block = program.block(sub_block_idx)
            grad_sub_block = program.create_block(parent_idx=sub_block_idx)
            backward_impl(target, sub_block, grad_sub_block, no_grad_set,
                          grad_info_map, callback)
            grad_sub_block_list.append(grad_sub_block)
        grad_op_desc = core.get_grad_op_desc(each_op.desc,
                                             no_grad_set[block.idx],
                                             grad_to_var, grad_sub_block_list)
        grad_op_descs.append(grad_op_desc)
    # grad_op_descs = [[op1_g1, op1_g2], [op2_g], ...]
    # flatten grad_op_descs
    grad_op_descs = [op for sublist in grad_op_descs for op in sublist]  # ?????

    pending_sum_ops = []
    var_rename_count = collections.defaultdict(int)
    var_inputs = collections.defaultdict(list)
    for pos, op_desc in enumerate(grad_op_descs):
        for var_name in op_desc.input_arg_names():
            if len(var_inputs[var_name]) > 1:
                pdb.set_trace()
                pending_sum_ops.append((core.OpDesc(
                    type="sum_op",
                    inputs=var_inputs[var_name],
                    output=[var_name],
                    attrs={}), pos))
                var_inputs[var_name] = [var_name]
        for var_name in op_desc.output_arg_names():
            if len(var_inputs[var_name]) == 0:
                # it's the first time we get the variable
                var_inputs[var_name] = [var_name]
            else:
                if len(var_inputs[var_name] == 1):
                    new_name = var_name + "@RENAME@" + \
                        str(var_rename_count[var_name])
                    var_rename_count[var_name] = var_rename_count[var_name] + 1
                    # rename original var_name
                    var_inputs[var_name][0] = new_name
                    rename_arg(grad_op_descs, var_name, new_name, 0, pos)
                    rename_arg(pending_sum_ops, var_name, new_name)

                new_name = var_name + "@RENAME@" + \
                    str(var_rename_count[var_name])
                var_rename_count[var_name] = var_rename_count[var_name] + 1
                op_desc.rename_output(var_name, new_name)
                var_inputs[var_name].append(new_name)
    for var_name, inputs in var_inputs.iteritems():
        if len(inputs) > 1:
            pdb.set_trace()
            pending_sum_ops.append((core.OpDesc("sum_op", {"X": inputs},
                                                {"Out": var_name}, {}),
                                    len(grad_op_descs)))
    # TODO: remove op in no grad set

    # 根据append的顺序可以看出pending_sum_ops一定是根据sum_op的插入位置排序的
    for p in reversed(pending_sum_ops):
        grad_op_descs.insert(p[1], p[0])
    # create new gradient variables in the target block desc
    for op_desc in grad_op_descs:
        for grad_var_name in op_desc.output_arg_names():
            grad_var_name = grad_var_name.encode("ascii")
            if target_block.desc.has_var(
                    grad_var_name) or grad_var_name == core.get_empty_var_name(
                    ):
                continue
            target_block.desc.var(grad_var_name)
            if not grad_to_var.has_key(grad_var_name):
                continue
            grad_info_map[grad_to_var[grad_var_name]] = (grad_var_name,
                                                         target_block)
    if target_block.idx == 0:
        grad_target_name = (target.name + "@GRAD")
        target_block.desc.var(grad_target_name)
        grad_op_descs.insert(
            0,
            core.OpDesc(u"fill_constant", {}, {
                u"Out": [unicode(grad_target_name, "ascii")]
            }, {u"shape": (1),
                u"value": 1.0,
                u"dtype": core.DataType.FP32}))
    # insert backward operators to target_block
    for op_desc in grad_op_descs:
        target_block.desc.append_allocated_op(op_desc)

    target_block.sync_with_cpp()


def append_backward_ops(loss, parameter_list=None, no_grad_set=None):
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

    if no_grad_set is None:
        no_grad_set = dict()
        program = loss.block.program
        assert isinstance(program, framework.Program)
        for block in program.blocks:
            assert isinstance(block, framework.Block)
            block_no_grad_set = set()
            for var in block.vars.itervalues():
                assert isinstance(var, framework.Variable)
                if var.stop_gradient:
                    block_no_grad_set.add(var.name)
            no_grad_set[block.idx] = block_no_grad_set

    grad_info_map = dict()
    root_block = loss.block.program.block(0)
    backward_impl(loss, root_block, root_block, no_grad_set, grad_info_map)
    pdb.set_trace()
    if parameter_list is not None:
        parameters = parameter_list
    else:
        params = loss.block.program.global_block().all_parameters()
        parameters = [param.name for param in params]
    params_and_grads = []
    for param in parameters:
        if param not in grad_info_map:
            raise ValueError("param %s is not in map" % param)
        grad_info = grad_info_map[param]
        grad_block = loss.block.program.block(grad_info[1])
        if not grad_block.has_var(grad_info[0]):
            raise ValueError("grad block[{0}] did not have grad var {1}".format(
                grad_info[1], grad_info[0]))
        # Get the param var from the global block
        param_var = loss.block.program.global_block().var(param)
        grad_var = grad_block.var(grad_info[0])
        if loss.block.has_var(grad_info[0]):
            params_and_grads.append((param_var, grad_var))
        else:
            params_and_grads.append((param_var, None))
    return params_and_grads
