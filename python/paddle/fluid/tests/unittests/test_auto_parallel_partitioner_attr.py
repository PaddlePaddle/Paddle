# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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


def get_var(op, main_program):
    varname = op.desc.input_arg_names()
    var = main_program.global_block().var(varname[0])
    return var


def get_input_var_dist_attr(op, main_program, dist_context):
    var = get_var(op, main_program)
    dist_attr = dist_context.get_tensor_distributed_attr_for_program(var)
    return dist_attr


def get_output_var_dist_attr(op, main_program, dist_context):
    var = get_var(op, main_program)
    dist_attr = dist_context.get_tensor_distributed_attr_for_program(var)
    return dist_attr


def check_equal_var_dist_attr(serial_op, dist_op, serial_main_prog,
                              dist_main_prog, serial_dist_attr, dist_attr):
    serial_var = get_var(serial_op, serial_main_prog)
    dist_var = get_var(serial_op, serial_main_prog)
    equal = True
    if serial_var.desc.id() == dist_var.desc.id() or \
        serial_dist_attr.get_process_mesh() != dist_attr.get_process_mesh() or \
        serial_dist_attr.is_parameter() != dist_attr.is_parameter() or \
        serial_dist_attr.get_dims_mapping() != dist_attr.get_dims_mapping():
        equal = False
    return equal


def check_equal_dist_op_attr(dist_context, dist_main_prog, serial_op, dist_ops,
                             dist_op_idx):
    equal = True
    # get serial op's process_mesh and impl_idx
    serial_op_dist_attr = dist_context.get_op_distributed_attr_for_program(
        serial_op)
    serial_process_mesh = serial_op_dist_attr.get_process_mesh()
    serial_impl_idx = serial_op_dist_attr.get_impl_idx()

    # check dist_attr between serial op and dist op
    for i in dist_op_idx:
        op_dist_attr = dist_context.get_op_distributed_attr_for_program(
            dist_ops[i])
        for in_varname in dist_ops[i].desc.input_arg_names():
            in_var = dist_main_prog.global_block().var(in_varname)
            tensor_dist_attr = dist_context.get_tensor_distributed_attr_for_program(
                in_var)
            tensor_dims_mapping = tensor_dist_attr.get_dims_mapping()
            in_var_dims_mapping = op_dist_attr.get_input_dims_mapping(
                in_varname)
            if tensor_dims_mapping != in_var_dims_mapping:
                equal = False
        for out_varname in dist_ops[i].desc.output_arg_names():
            out_var = dist_main_prog.global_block().var(out_varname)
            tensor_dist_attr = dist_context.get_tensor_distributed_attr_for_program(
                out_var)
            tensor_dims_mapping = tensor_dist_attr.get_dims_mapping()
            out_var_dims_mapping = op_dist_attr.get_output_dims_mapping(
                out_varname)
            if tensor_dims_mapping != out_var_dims_mapping:
                equal = False

        dist_op_process_mesh = op_dist_attr.get_process_mesh()
        dist_op_impl_idx = op_dist_attr.get_impl_idx()
        if serial_op.desc.id() == dist_ops[i].desc.id() or \
            serial_process_mesh != dist_op_process_mesh or \
            serial_impl_idx != dist_op_impl_idx:
            equal = False

    return equal


def distributed_attr_check_for_dist_op(serial_main_prog, dist_main_prog,
                                       dist_context, serial_op_idx,
                                       dist_op_idx):

    equal = True
    serial_ops = serial_main_prog.global_block().ops
    dist_ops = dist_main_prog.global_block().ops

    for i in range(len(serial_op_idx)):
        serial_op = serial_ops[serial_op_idx[i]]
        dist_op_0 = dist_ops[dist_op_idx[i][0]]
        if dist_op_0.type == "c_identity":
            # serial op input's dist_attr
            serial_in_dist_attr = get_input_var_dist_attr(
                serial_op, serial_main_prog, dist_context)
            # c_identity output's(new var) dist_attr
            identity_out_dist_attr = get_output_var_dist_attr(
                dist_op_0, dist_main_prog, dist_context)
            # check var dist_attr
            equal = check_equal_var_dist_attr(
                serial_op, dist_op_0, serial_main_prog, dist_main_prog,
                serial_in_dist_attr, identity_out_dist_attr)
        else:
            # serial op output's dist_attr
            serial_out_dist_attr = get_output_var_dist_attr(
                serial_op, serial_main_prog, dist_context)
            # dist op output's(new var) dist_attr
            out_dist_attr = get_output_var_dist_attr(dist_op_0, dist_main_prog,
                                                     dist_context)
            # check var dist_attr
            equal = check_equal_var_dist_attr(
                serial_op, dist_op_0, serial_main_prog, dist_main_prog,
                serial_out_dist_attr, out_dist_attr)

        # check op's dist_attr 
        equal = check_equal_dist_op_attr(dist_context, dist_main_prog,
                                         serial_op, dist_ops, dist_op_idx[i])

    return equal
