# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import json
import sys

SAME = 0

INPUTS = "Inputs"
OUTPUTS = "Outputs"
ATTRS = "Attrs"

ADD = "Add"
DELETE = "Delete"
CHANGE = "Change"

DUPLICABLE = "duplicable"
INTERMEDIATE = "intermediate"
DISPENSABLE = "dispensable"

TYPE = "type"
GENERATED = "generated"
DEFAULT_VALUE = "default_value"

error = False


def diff_vars(origin_vars, new_vars):
    global error
    var_error = False
    var_changed_error_massage = {}
    var_added_error_massage = []
    var_deleted_error_massage = []

    common_vars_name = set(origin_vars.keys()) & set(new_vars.keys())
    vars_name_only_in_origin = set(origin_vars.keys()) - set(new_vars.keys())
    vars_name_only_in_new = set(new_vars.keys()) - set(origin_vars.keys())

    for var_name in common_vars_name:
        if cmp(origin_vars.get(var_name), new_vars.get(var_name)) == SAME:
            continue
        else:
            error, var_error = True, True
            var_changed_error_massage[var_name] = {}
            for arg_name in origin_vars.get(var_name):
                new_arg_value = new_vars.get(var_name, {}).get(arg_name)
                origin_arg_value = origin_vars.get(var_name, {}).get(arg_name)
                if new_arg_value != origin_arg_value:
                    var_changed_error_massage[var_name][arg_name] = (
                        origin_arg_value, new_arg_value)

    for var_name in vars_name_only_in_origin:
        error, var_error = True, True
        var_deleted_error_massage.append(var_name)

    for var_name in vars_name_only_in_new:
        if not new_vars.get(var_name).get(DISPENSABLE):
            error, var_error = True, True
            var_added_error_massage.append(var_name)

    var_diff_message = {}
    if var_added_error_massage:
        var_diff_message[ADD] = var_added_error_massage
    if var_changed_error_massage:
        var_diff_message[CHANGE] = var_changed_error_massage
    if var_deleted_error_massage:
        var_diff_message[DELETE] = var_deleted_error_massage

    return var_error, var_diff_message


def diff_attr(ori_attrs, new_attrs):
    global error
    attr_error = False

    attr_changed_error_massage = {}
    attr_added_error_massage = []
    attr_deleted_error_massage = []

    common_attrs = set(ori_attrs.keys()) & set(new_attrs.keys())
    attrs_only_in_origin = set(ori_attrs.keys()) - set(new_attrs.keys())
    attrs_only_in_new = set(new_attrs.keys()) - set(ori_attrs.keys())

    for attr_name in common_attrs:
        if cmp(ori_attrs.get(attr_name), new_attrs.get(attr_name)) == SAME:
            continue
        else:
            error, attr_error = True, True
            attr_changed_error_massage[attr_name] = {}
            for arg_name in ori_attrs.get(attr_name):
                new_arg_value = new_attrs.get(attr_name, {}).get(arg_name)
                origin_arg_value = ori_attrs.get(attr_name, {}).get(arg_name)
                if new_arg_value != origin_arg_value:
                    attr_changed_error_massage[attr_name][arg_name] = (
                        origin_arg_value, new_arg_value)

    for attr_name in attrs_only_in_origin:
        error, attr_error = True, True
        attr_deleted_error_massage.append(attr_name)

    for attr_name in attrs_only_in_new:
        if new_attrs.get(attr_name).get(DEFAULT_VALUE) == None:
            error, attr_error = True, True
            attr_added_error_massage.append(attr_name)

    attr_diff_message = {}
    if attr_added_error_massage:
        attr_diff_message[ADD] = attr_added_error_massage
    if attr_changed_error_massage:
        attr_diff_message[CHANGE] = attr_changed_error_massage
    if attr_deleted_error_massage:
        attr_diff_message[DELETE] = attr_deleted_error_massage

    return attr_error, attr_diff_message


def compare_op_desc(origin_op_desc, new_op_desc):
    origin = json.loads(origin_op_desc)
    new = json.loads(new_op_desc)
    error_message = {}
    if cmp(origin_op_desc, new_op_desc) == SAME:
        return error_message

    for op_type in origin:

        # no need to compare if the operator is deleted
        if op_type not in new:
            continue

        origin_info = origin.get(op_type, {})
        new_info = new.get(op_type, {})

        origin_inputs = origin_info.get(INPUTS, {})
        new_inputs = new_info.get(INPUTS, {})
        ins_error, ins_diff = diff_vars(origin_inputs, new_inputs)

        origin_outputs = origin_info.get(OUTPUTS, {})
        new_outputs = new_info.get(OUTPUTS, {})
        outs_error, outs_diff = diff_vars(origin_outputs, new_outputs)

        origin_attrs = origin_info.get(ATTRS, {})
        new_attrs = new_info.get(ATTRS, {})
        attrs_error, attrs_diff = diff_attr(origin_attrs, new_attrs)

        if ins_error:
            error_message.setdefault(op_type, {})[INPUTS] = ins_diff
        if outs_error:
            error_message.setdefault(op_type, {})[OUTPUTS] = outs_diff
        if attrs_error:
            error_message.setdefault(op_type, {})[ATTRS] = attrs_diff

    return error_message


def print_error_message(error_message):
    print("Op desc error for the changes of Inputs/Outputs/Attrs of OPs:\n")
    for op_name in error_message:
        print("For OP '{}':".format(op_name))

        # 1. print inputs error message
        Inputs_error = error_message.get(op_name, {}).get(INPUTS, {})
        for name in Inputs_error.get(ADD, {}):
            print(" * The added Input '{}' is not dispensable.".format(name))

        for name in Inputs_error.get(DELETE, {}):
            print(" * The Input '{}' is deleted.".format(name))

        for name in Inputs_error.get(CHANGE, {}):
            changed_args = Inputs_error.get(CHANGE, {}).get(name, {})
            for arg in changed_args:
                ori_value, new_value = changed_args.get(arg)
                print(
                    " * The arg '{}' of Input '{}' is changed: from '{}' to '{}'.".
                    format(arg, name, ori_value, new_value))

        # 2. print outputs error message
        Outputs_error = error_message.get(op_name, {}).get(OUTPUTS, {})
        for name in Outputs_error.get(ADD, {}):
            print(" * The added Output '{}' is not dispensable.".format(name))

        for name in Outputs_error.get(DELETE, {}):
            print(" * The Output '{}' is deleted.".format(name))

        for name in Outputs_error.get(CHANGE, {}):
            changed_args = Outputs_error.get(CHANGE, {}).get(name, {})
            for arg in changed_args:
                ori_value, new_value = changed_args.get(arg)
                print(
                    " * The arg '{}' of Output '{}' is changed: from '{}' to '{}'.".
                    format(arg, name, ori_value, new_value))

        # 3. print attrs error message
        attrs_error = error_message.get(op_name, {}).get(ATTRS, {})
        for name in attrs_error.get(ADD, {}):
            print(" * The added attr '{}' doesn't set default value.".format(
                name))

        for name in attrs_error.get(DELETE, {}):
            print(" * The attr '{}' is deleted.".format(name))

        for name in attrs_error.get(CHANGE, {}):
            changed_args = attrs_error.get(CHANGE, {}).get(name, {})
            for arg in changed_args:
                ori_value, new_value = changed_args.get(arg)
                print(
                    " * The arg '{}' of attr '{}' is changed: from '{}' to '{}'.".
                    format(arg, name, ori_value, new_value))


def print_repeat_process():
    print(
        "Tips:"
        " If you want to repeat the process, please follow these steps:\n"
        "\t1. Compile and install paddle from develop branch \n"
        "\t2. Run: python tools/print_op_desc.py  > OP_DESC_DEV.spec \n"
        "\t3. Compile and install paddle from PR branch \n"
        "\t4. Run: python tools/print_op_desc.py  > OP_DESC_PR.spec \n"
        "\t5. Run: python tools/check_op_desc.py OP_DESC_DEV.spec OP_DESC_PR.spec"
    )


if len(sys.argv) == 3:
    '''
    Compare op_desc files generated by branch DEV and branch PR.
    And print error message.
    '''
    with open(sys.argv[1], 'r') as f:
        origin_op_desc = f.read()

    with open(sys.argv[2], 'r') as f:
        new_op_desc = f.read()

    error_message = compare_op_desc(origin_op_desc, new_op_desc)
    if error:
        print("-" * 30)
        print_error_message(error_message)
        print("-" * 30)
else:
    print("Usage: python check_op_desc.py OP_DESC_DEV.spec OP_DESC_PR.spec")
