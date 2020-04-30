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
import paddle
import paddle.fluid as fluid


def cast_fp16_to_fp32(i, o, prog):
    prog.global_block().append_op(
        type="cast",
        inputs={"X": i},
        outputs={"Out": o},
        attrs={
            "in_dtype": fluid.core.VarDesc.VarType.FP16,
            "out_dtype": fluid.core.VarDesc.VarType.FP32
        })


def cast_fp32_to_fp16(i, o, prog):
    prog.global_block().append_op(
        type="cast",
        inputs={"X": i},
        outputs={"Out": o},
        attrs={
            "in_dtype": fluid.core.VarDesc.VarType.FP32,
            "out_dtype": fluid.core.VarDesc.VarType.FP16
        })


def copy_to_master_param(p, block):
    v = block.vars.get(p.name, None)
    if v is None:
        raise ValueError("no param name %s found!" % p.name)
    new_p = fluid.framework.Parameter(
        block=block,
        shape=v.shape,
        dtype=fluid.core.VarDesc.VarType.FP32,
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
    master_params_grads = []
    tmp_role = main_prog._current_role
    OpRole = fluid.core.op_proto_and_checker_maker.OpRole
    main_prog._current_role = OpRole.Backward
    for p, g in params_grads:
        # create master parameters
        master_param = copy_to_master_param(p, main_prog.global_block())
        startup_master_param = startup_prog.global_block()._clone_variable(
            master_param)
        startup_p = startup_prog.global_block().var(p.name)
        cast_fp16_to_fp32(startup_p, startup_master_param, startup_prog)
        # cast fp16 gradients to fp32 before apply gradients
        if g.name.find("layer_norm") > -1:
            if loss_scaling > 1:
                scaled_g = g / float(loss_scaling)
            else:
                scaled_g = g
            master_params_grads.append([p, scaled_g])
            continue
        master_grad = fluid.layers.cast(g, "float32")
        if loss_scaling > 1:
            master_grad = master_grad / float(loss_scaling)
        master_params_grads.append([master_param, master_grad])
    main_prog._current_role = tmp_role
    return master_params_grads


def master_param_to_train_param(master_params_grads, params_grads, main_prog):
    for idx, m_p_g in enumerate(master_params_grads):
        train_p, _ = params_grads[idx]
        if train_p.name.find("layer_norm") > -1:
            continue
        with main_prog._optimized_guard([m_p_g[0], m_p_g[1]]):
            cast_fp32_to_fp16(m_p_g[0], train_p, main_prog)
