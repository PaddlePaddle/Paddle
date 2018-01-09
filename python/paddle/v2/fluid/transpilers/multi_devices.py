from ..framework import Variable, Program, Block, Operator, unique_name
from .. import core

__all__ = ['transpile_to_multi_devices']


def _vars_to_names_(vars):
    if vars is None:
        vars = []
    for each_var in vars:
        if isinstance(each_var, Variable):
            yield str(each_var.name)
        elif isinstance(each_var, basestring):
            yield str(each_var)
        else:
            raise TypeError("Vars should be Variable or basestring")


def _get_inputs_and_params_names_(block):
    outputs = set()
    inputs_and_params = set()
    for op in block.ops:
        assert isinstance(op, Operator)
        for i_name in op.input_names:
            for i_var_name in op.input(i_name):
                if i_var_name not in outputs:
                    inputs_and_params.add(str(i_var_name))

        for o_name in op.output_names:
            for o_var_name in op.output(o_name):
                outputs.add(str(o_var_name))
    return inputs_and_params


def _move_copy_var_to_block_(var_names, src_block, dst_block, remove=True):
    for p_name in var_names:
        p_var = src_block.vars.get(p_name, None)
        if p_var is None:
            raise ValueError(
                "cannot move parameter up to dst_block. Variable %s not found",
                p_name)
        assert p_name not in dst_block.vars
        dst_block.clone_var(p_var)
        if remove:
            src_block.remove_var(p_var)


def _move_var_to_block_(var_names, src_block, dst_block):
    _move_copy_var_to_block_(var_names, src_block, dst_block, remove=True)


def _copy_var_to_block_(var_names, src_block, dst_block):
    _move_copy_var_to_block_(var_names, src_block, dst_block, remove=False)


def _var_lists_(var_names, block):
    return [block.var(var_name) for var_name in var_names]


def transpile_to_multi_devices(program,
                               input_vars,
                               output_vars=None,
                               block_id=0,
                               device_type='CPU',
                               device_count=0):
    # Get original block
    src_block = program.block(block_id)
    assert isinstance(src_block, Block)

    # Create new program
    new_prog = program.clone()
    assert isinstance(new_prog, Program)

    # Create the sub_block of parallel.for
    sub_block = new_prog.create_block(src_block.parent_idx)
    # sub_block of parallel.for is as same as original block
    sub_block.copy_from(src_block)

    # Get the dst_block and clear it
    dst_block = new_prog.block(block_id)
    assert isinstance(dst_block, Block)
    dst_block.clear()

    # Add GetPlaces op
    places = dst_block.create_var(
        name=unique_name('multidev_transpiler_get_place_out'))
    dst_block.append_op(
        type='get_places',
        outputs={"Out": [places]},
        attrs={
            "device_type": device_type,
            'device_count': device_count,
        })

    # Get Input Var names, Param names of sub block
    input_vars = set(_vars_to_names_(input_vars))
    inputs_and_param_names = _get_inputs_and_params_names_(src_block)
    if inputs_and_param_names.issubset(input_vars):
        raise ValueError(
            "Input vars is not a subset of all inputs and parameters in block "
            "{0}.\n {1} \\notin {2}".format(
                str(block_id), str(input_vars), str(inputs_and_param_names)))
    param_names = inputs_and_param_names - input_vars

    # move param from sub_block to current block
    _move_var_to_block_(
        var_names=param_names, src_block=sub_block, dst_block=dst_block)

    # Copy input variable to current blocks
    _copy_var_to_block_(
        var_names=input_vars, src_block=sub_block, dst_block=dst_block)

    output_vars = set(_vars_to_names_(output_vars))
    # Copy output variable to current block
    _copy_var_to_block_(
        var_names=output_vars, src_block=sub_block, dst_block=dst_block)

    scopes = dst_block.create_var(type=core.VarDesc.VarType.STEP_SCOPES)
    outs = _var_lists_(output_vars, dst_block)
    dst_block.append_op(
        type='parallel_do',
        inputs={
            'inputs': _var_lists_(input_vars, dst_block),
            'parameters': _var_lists_(param_names, dst_block),
            'places': [places]
        },
        outputs={'outputs': outs,
                 'parallel_scopes': scopes},
        attrs={'sub_block': sub_block})
    new_prog.current_block_idx = block_id
    return new_prog, outs
