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
    layers.reshape(prev_loss_scaling, [1, 1], inplace=True)
    update_loss_scaling_control = layers.IfElse(is_overall_finite)
    # increase loss scaling
    with update_loss_scaling_control.true_block():
        true_incr_every_n_steps = \
                update_loss_scaling_control.input(incr_every_n_steps)
        true_num_good_steps = update_loss_scaling_control.input(num_good_steps)
        should_incr_loss_scaling = layers.less_than(true_incr_every_n_steps,
                                                    true_num_good_steps + 1)
        incr_loss_scaling_control = layers.IfElse(should_incr_loss_scaling)
        # increase loss scaling when accumulated num_good_steps 
        # larger than incr_every_n_steps
        with incr_loss_scaling_control.true_block():
            loss_scaling = incr_loss_scaling_control.input(prev_loss_scaling)
            num_good_steps_tmp = incr_loss_scaling_control.input(num_good_steps)
            num_bad_steps_tmp = incr_loss_scaling_control.input(num_bad_steps)
            new_loss_scaling = loss_scaling * incr_ratio
            is_finite = layers.isfinite(new_loss_scaling)
            is_finite_for_reshape = layers.cast(is_finite, 'int32')
            is_finite_for_reshape = layers.reshape(
                is_finite_for_reshape, [1, 1], inplace=True)
            loss_scaling_is_finite = layers.cast(is_finite_for_reshape, 'bool')
            apply_incr_loss_scaling_control = \
                    layers.IfElse(loss_scaling_is_finite)
            # check whether the increased loss scaling is inf
            with apply_incr_loss_scaling_control.true_block():
                apply_incr_loss_scaling_control.output(new_loss_scaling)
            with apply_incr_loss_scaling_control.false_block():
                false_new_loss_scaling = \
                        apply_incr_loss_scaling_control.input(prev_loss_scaling)
                apply_incr_loss_scaling_control.output(false_new_loss_scaling)
            updated_loss_scaling, = apply_incr_loss_scaling_control()
            num_good_steps_tmp *= 0
            num_bad_steps_tmp *= 0
            incr_loss_scaling_control.output(
                updated_loss_scaling, num_good_steps_tmp, num_bad_steps_tmp)
        # maintain the loss scaling
        with incr_loss_scaling_control.false_block():
            loss_scaling = incr_loss_scaling_control.input(prev_loss_scaling)
            num_good_steps_tmp = incr_loss_scaling_control.input(num_good_steps)
            num_bad_steps_tmp = incr_loss_scaling_control.input(num_bad_steps)
            num_good_steps_tmp += 1
            num_bad_steps_tmp *= 0
            incr_loss_scaling_control.output(loss_scaling, num_good_steps_tmp,
                                             num_bad_steps_tmp)
        new_loss_scaling, new_num_good_steps, new_num_bad_steps = \
                incr_loss_scaling_control()
        update_loss_scaling_control.output(new_loss_scaling, new_num_good_steps,
                                           new_num_bad_steps)
    # decrease loss scaling
    with update_loss_scaling_control.false_block():
        false_decr_every_n_nan_or_inf = \
                update_loss_scaling_control.input(decr_every_n_nan_or_inf)
        false_num_bad_steps = update_loss_scaling_control.input(num_bad_steps)
        should_decr_loss_scaling = layers.less_than(
            false_decr_every_n_nan_or_inf, false_num_bad_steps + 1)
        decr_loss_scaling_control = layers.IfElse(should_decr_loss_scaling)
        # decrease loss scaling when accumulated num_bad_steps larger than
        # decr_every_n_nan_or_inf
        with decr_loss_scaling_control.true_block():
            loss_scaling = decr_loss_scaling_control.input(prev_loss_scaling)
            num_good_steps_tmp = decr_loss_scaling_control.input(num_good_steps)
            num_bad_steps_tmp = decr_loss_scaling_control.input(num_bad_steps)
            new_loss_scaling = loss_scaling * decr_ratio
            static_loss_scaling = \
                layers.fill_constant(shape=[1,1], dtype='float32', value=1.0)
            less_than_one = layers.less_than(new_loss_scaling,
                                             static_loss_scaling)
            apply_decr_loss_scaling_control = layers.IfElse(less_than_one)
            # check whether the decreased loss scaling less than one
            with apply_decr_loss_scaling_control.true_block():
                static_loss_scaling_tmp = \
                    apply_decr_loss_scaling_control.input(static_loss_scaling)
                apply_decr_loss_scaling_control.output(static_loss_scaling_tmp)
            with apply_decr_loss_scaling_control.false_block():
                apply_decr_loss_scaling_control.output(new_loss_scaling)
            updated_loss_scaling, = apply_decr_loss_scaling_control()
            num_good_steps_tmp *= 0
            num_bad_steps_tmp *= 0
            decr_loss_scaling_control.output(
                updated_loss_scaling, num_good_steps_tmp, num_bad_steps_tmp)
        # maintain the loss scaling
        with decr_loss_scaling_control.false_block():
            loss_scaling = decr_loss_scaling_control.input(prev_loss_scaling)
            num_good_steps_tmp = decr_loss_scaling_control.input(num_good_steps)
            num_bad_steps_tmp = decr_loss_scaling_control.input(num_bad_steps)
            num_good_steps_tmp *= 0
            num_bad_steps_tmp += 1
            decr_loss_scaling_control.output(loss_scaling, num_good_steps_tmp,
                                             num_bad_steps_tmp)

        new_loss_scaling, new_num_good_steps, new_num_bad_steps = \
            decr_loss_scaling_control()
        update_loss_scaling_control.output(new_loss_scaling, new_num_good_steps,
                                           new_num_bad_steps)
    # update the loss scaling, num_good_steps, num_bad_steps
    new_loss_scaling, new_num_good_steps, new_num_bad_steps = \
            update_loss_scaling_control()
    layers.assign(new_loss_scaling, prev_loss_scaling)
    layers.reshape(prev_loss_scaling, [1], inplace=True)
    layers.assign(new_num_good_steps, num_good_steps)
    layers.assign(new_num_bad_steps, num_bad_steps)
