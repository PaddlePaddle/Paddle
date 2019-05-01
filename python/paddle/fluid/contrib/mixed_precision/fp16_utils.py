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
                if loss_scaling > 1:
                    scaled_g = g / float(loss_scaling)
                else:
                    scaled_g = g
                master_params_grads.append([p, scaled_g])
                continue
            master_grad = layers.cast(x=g, dtype="float32")
            if loss_scaling > 1:
                master_grad = master_grad / float(loss_scaling)
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
