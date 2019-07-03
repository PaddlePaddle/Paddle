#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from ... import core
from ... import layers
from ... import framework


def append_cast_op(i, o, prog):
    """
    Append a cast op in a given Program to cast input `i` to data type `o.dtype`.

    Args:
        i (Variable): The input Variable.
        o (Variable): The output Variable.
        prog (Program): The Program to append cast op.
    """
    prog.global_block().append_op(
        type="cast",
        inputs={"X": i},
        outputs={"Out": o},
        attrs={"in_dtype": i.dtype,
               "out_dtype": o.dtype})


def copy_to_master_param(p, block):
    """
    New a master parameter for the input parameter, and they two share the same
    attributes except the data type.

    Args:
        p(Parameter): The input parameter in float16.
        block(Program): The block in which the parameter is.
    """
    v = block.vars.get(p.name, None)
    if v is None:
        raise ValueError("no param name %s found!" % p.name)
    new_p = framework.Parameter(
        block=block,
        shape=v.shape,
        dtype=core.VarDesc.VarType.FP32,
        type=v.type,
        lod_level=v.lod_level,
        stop_gradient=p.stop_gradient,
        trainable=p.trainable,
        optimize_attr=p.optimize_attr,
        regularizer=p.regularizer,
        gradient_clip_attr=p.gradient_clip_attr,
        error_clip=p.error_clip,
        name=v.name + ".master")
    return new_p


def create_master_params_grads(params_grads, main_prog, startup_prog,
                               loss_scaling):
    """ 
    Create master parameters and gradients in float32 from params and grads 
    in float16.

    Args:
        params_grads (list): A list of tuple (parameter, gradient) in float32.
        main_prog (Program): The main program for training.
        startup_prog (Program): The startup program to initialize all parameters.
        loss_scaling (float): The factor to scale loss and gradients.

    Returns:
        A list of master parameters and gradients. 
    """
    master_params_grads = []
    with main_prog._backward_role_guard():
        for p, g in params_grads:
            # create master parameters
            master_param = copy_to_master_param(p, main_prog.global_block())
            startup_master_param = startup_prog.global_block()._clone_variable(
                master_param)
            startup_p = startup_prog.global_block().var(p.name)
            # fp16 -> fp32
            append_cast_op(startup_p, startup_master_param, startup_prog)
            # cast fp16 gradients to fp32 before apply gradients
            if g.name.find("batch_norm") > -1:
                scaled_g = g / loss_scaling
                master_params_grads.append([p, scaled_g])
                continue
            master_grad = layers.cast(x=g, dtype="float32")
            master_grad = master_grad / loss_scaling
            master_params_grads.append([master_param, master_grad])

    return master_params_grads


def master_param_to_train_param(master_params_grads, params_grads, main_prog):
    """ 
    Convert master master parameters and gradients in float32 to parameters and 
    gradients in float16 for forward computation.

    Args:
        master_params_grads (list): A list of master parameters and gradients in 
                                   float32.
        params_grads (list): A list of parameters and gradients in float16.
        main_prog (list): The main program for execution.
    """
    for idx, m_p_g in enumerate(master_params_grads):
        train_p, _ = params_grads[idx]
        if train_p.name.find("batch_norm") > -1:
            continue
        with main_prog._optimized_guard([m_p_g[0], m_p_g[1]]):
            # fp32 -> fp16
            append_cast_op(m_p_g[0], train_p, main_prog)


def _rename_arg(op, old_name, new_name):
    """
    If an op has old_name input and output, rename these input 
    args new_name.

    Args:
        op (Operator): Current operator.
        old_name (str): The old name of input args.
        new_name (str): The new name of input args.
    """
    op_desc = op.desc
    if isinstance(op_desc, tuple):
        op_desc = op_desc[0]
    op_desc._rename_input(old_name, new_name)
    op_desc._rename_output(old_name, new_name)


def _dtype_to_str(dtype):
    """
    Convert specific variable type to its corresponding string.

    Args:
        dtype (VarType): Variable type.
    """
    if dtype == core.VarDesc.VarType.FP16:
        return 'fp16'
    else:
        return 'fp32'


def _insert_cast_op(block, op, idx, src_dtype, dest_dtype):
    """
    Insert cast op and rename args of input and output.

    Args:
        block (Program): The block in which the operator is.
        op (Operator): The operator to insert cast op.
        idx (int): The index of current operator.
        src_dtype (VarType): The input variable dtype of cast op.
        desr_dtype (VarType): The output variable dtype of cast op.

    Returns:
        num_cast_op (int): The number of cast ops that have been inserted.
    """
    num_cast_ops = 0
    valid_types = [
        core.VarDesc.VarType.LOD_TENSOR, core.VarDesc.VarType.SELECTED_ROWS,
        core.VarDesc.VarType.LOD_TENSOR_ARRAY
    ]
    for in_name in op.input_names:
        for in_var_name in op.input(in_name):
            in_var = block.var(in_var_name)
            if in_var.type not in valid_types:
                continue
            if in_var.dtype == src_dtype:
                out_var = block.create_var(
                    name=in_var.name + \
                            '.cast_' + _dtype_to_str(dest_dtype),
                    dtype=dest_dtype,
                    persistable=False,
                    stop_gradient=False)
                block._insert_op(
                    idx,
                    type="cast",
                    inputs={"X": in_var},
                    outputs={"Out": out_var},
                    attrs={
                        "in_dtype": in_var.dtype,
                        "out_dtype": out_var.dtype
                    })
                num_cast_ops += 1
                _rename_arg(op, in_var.name, out_var.name)
            else:
                if op.has_attr('in_dtype'):
                    op._set_attr('in_dtype', dest_dtype)
    if src_dtype == core.VarDesc.VarType.FP16:
        for out_name in op.output_names:
            for out_var_name in op.output(out_name):
                out_var = block.var(out_var_name)
                if out_var.type not in valid_types:
                    continue
                if out_var.dtype == core.VarDesc.VarType.FP16:
                    out_var.desc.set_dtype(core.VarDesc.VarType.FP32)
                    if op.has_attr('out_dtype'):
                        op._set_attr('out_dtype', core.VarDesc.VarType.FP32)
    return num_cast_ops


def find_true_prev_op(ops, var_name):
    for op in ops:
        for out_name in op.output_names:
            for out_var_name in op.output(out_name):
                if out_var_name == var_name:
                    return op


def rewrite_program(main_prog, amp_lists):
    """
    Traverse all ops in current block and insert cast op according to 
    which set current op belongs to.

    1. When an op belongs to the black list, add it to black set
    2. When an op belongs to the white list, add it to white set
    3. When an op belongs to the gray list. If one 
       of its inputs is the output of black set op or black list op, 
       add it to black set. If all of its previous ops are not black 
       op and one of its inputs is the output of white set op or 
       white list op, add it to white set.
    4. When an op isn't in the lists, add it to black op set.
    5. Add necessary cast ops to make sure that black set op will be 
       computed in fp32 mode, while white set op will be computed in 
       fp16 mode.

    Args:
        main_prog (Program): The main program for training.
    """
    block = main_prog.global_block()
    ops = block.ops
    white_op_set = set()
    black_op_set = set()
    for i in range(len(ops)):
        op = ops[i]
        if op.type in amp_lists.black_list:
            black_op_set.add(op)
        elif op.type in amp_lists.white_list:
            white_op_set.add(op)
        elif op.type in amp_lists.gray_list:
            is_black_op = False
            is_white_op = False
            for in_name in op.input_names:
                # if this op has inputs
                if in_name:
                    for in_var_name in op.input(in_name):
                        in_var = block.var(in_var_name)
                        # this in_var isn't the output of other op
                        if in_var.op is None:
                            continue
                        if in_var.op is op:
                            prev_op = find_true_prev_op(ops, in_var_name)
                        else:
                            prev_op = in_var.op
                        # if it's one of inputs
                        if prev_op in black_op_set or \
                                prev_op.type in amp_lists.black_list:
                            is_black_op = True
                        if prev_op in white_op_set or \
                                prev_op.type in amp_lists.white_list:
                            is_white_op = True
            if is_black_op:
                black_op_set.add(op)
            elif is_white_op:
                white_op_set.add(op)
            else:
                pass
        else:
            # For numerical safe, we apply fp32 computation on ops that
            # are not determined which list they should stay.
            black_op_set.add(op)

    idx = 0
    while idx < len(ops):
        op = ops[idx]
        num_cast_ops = 0
        if op in black_op_set:
            num_cast_ops = _insert_cast_op(block, op, idx,
                                           core.VarDesc.VarType.FP16,
                                           core.VarDesc.VarType.FP32)
        elif op in white_op_set:
            num_cast_ops = _insert_cast_op(block, op, idx,
                                           core.VarDesc.VarType.FP32,
                                           core.VarDesc.VarType.FP16)
        else:
            pass

        idx += num_cast_ops + 1


def update_loss_scaling(is_overall_finite, prev_loss_scaling, num_good_steps,
                        num_bad_steps, incr_every_n_steps,
                        decr_every_n_nan_or_inf, incr_ratio, decr_ratio):
    """
    Update loss scaling according to overall gradients. If all gradients is 
    finite after incr_every_n_steps, loss scaling will increase by incr_ratio. 
    Otherwisw, loss scaling will decrease by decr_ratio after 
    decr_every_n_nan_or_inf steps and each step some gradients are infinite.

    Args:
        is_overall_finite (Variable): A boolean variable indicates whether 
                                     all gradients are finite.
        prev_loss_scaling (Variable): Previous loss scaling.
        num_good_steps (Variable): A variable accumulates good steps in which 
                                   all gradients are finite.
        num_bad_steps (Variable): A variable accumulates bad steps in which 
                                  some gradients are infinite.
        incr_every_n_steps (Variable): A variable represents increasing loss 
                                       scaling every n consecutive steps with 
                                       finite gradients.
        decr_every_n_nan_or_inf (Variable): A variable represents decreasing 
                                            loss scaling every n accumulated 
                                            steps with nan or inf gradients.
        incr_ratio(float): The multiplier to use when increasing the loss 
                           scaling.
        decr_ratio(float): The less-than-one-multiplier to use when decreasing 
                           loss scaling.
    """
    zero_steps = layers.fill_constant(shape=[1], dtype='int32', value=0)
    with layers.Switch() as switch:
        with switch.case(is_overall_finite):
            should_incr_loss_scaling = layers.less_than(incr_every_n_steps,
                                                        num_good_steps + 1)
            with layers.Switch() as switch1:
                with switch1.case(should_incr_loss_scaling):
                    new_loss_scaling = prev_loss_scaling * incr_ratio
                    loss_scaling_is_finite = layers.isfinite(new_loss_scaling)
                    with layers.Switch() as switch2:
                        with switch2.case(loss_scaling_is_finite):
                            layers.assign(new_loss_scaling, prev_loss_scaling)
                        with switch2.default():
                            pass
                    layers.assign(zero_steps, num_good_steps)
                    layers.assign(zero_steps, num_bad_steps)

                with switch1.default():
                    layers.increment(num_good_steps)
                    layers.assign(zero_steps, num_bad_steps)

        with switch.default():
            should_decr_loss_scaling = layers.less_than(decr_every_n_nan_or_inf,
                                                        num_bad_steps + 1)
            with layers.Switch() as switch3:
                with switch3.case(should_decr_loss_scaling):
                    new_loss_scaling = prev_loss_scaling * decr_ratio
                    static_loss_scaling = \
                        layers.fill_constant(shape=[1],
                                             dtype='float32',
                                             value=1.0)
                    less_than_one = layers.less_than(new_loss_scaling,
                                                     static_loss_scaling)
                    with layers.Switch() as switch4:
                        with switch4.case(less_than_one):
                            layers.assign(static_loss_scaling,
                                          prev_loss_scaling)
                        with switch4.default():
                            layers.assign(new_loss_scaling, prev_loss_scaling)
                    layers.assign(zero_steps, num_good_steps)
                    layers.assign(zero_steps, num_bad_steps)
                with switch3.default():
                    layers.assign(zero_steps, num_good_steps)
                    layers.increment(num_bad_steps)
