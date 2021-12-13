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
from paddle.utils import OpLastCheckpointChecker
from paddle.fluid.core import OpUpdateType

INPUTS = "Inputs"
OUTPUTS = "Outputs"
ATTRS = "Attrs"

# The constant `ADD` means that an item has been added. In particular,
# we use `ADD_WITH_DEFAULT` to mean adding attributes with default
# attributes, and `ADD_DISPENSABLE` to mean adding optional inputs or
# outputs.
ADD_WITH_DEFAULT = "Add_with_default"
ADD_DISPENSABLE = "Add_dispensable"
ADD = "Add"

DELETE = "Delete"
CHANGE = "Change"

DUPLICABLE = "duplicable"
INTERMEDIATE = "intermediate"
DISPENSABLE = "dispensable"

TYPE = "type"
GENERATED = "generated"
DEFAULT_VALUE = "default_value"

# add_with_extra, add_with_quant and add_with_def
EXTRA = "extra"
QUANT = "quant"
DEF = "def"

error = False

version_update_map = {
    INPUTS: {
        ADD: OpUpdateType.kNewInput,
    },
    OUTPUTS: {
        ADD: OpUpdateType.kNewOutput,
    },
    ATTRS: {
        ADD: OpUpdateType.kNewAttr,
        CHANGE: OpUpdateType.kModifyAttr,
    },
}


def diff_vars(origin_vars, new_vars):
    global error
    var_error = False
    var_changed_error_massage = {}
    var_add_massage = []
    var_add_dispensable_massage = []
    var_deleted_error_massage = []

    var_add_quant_message = []
    var_add_def_message = []

    common_vars_name = set(origin_vars.keys()) & set(new_vars.keys())
    vars_name_only_in_origin = set(origin_vars.keys()) - set(new_vars.keys())
    vars_name_only_in_new = set(new_vars.keys()) - set(origin_vars.keys())

    for var_name in common_vars_name:
        if origin_vars.get(var_name) == new_vars.get(var_name):
            continue
        else:
            error, var_error = True, True
            for arg_name in origin_vars.get(var_name):
                new_arg_value = new_vars.get(var_name, {}).get(arg_name)
                origin_arg_value = origin_vars.get(var_name, {}).get(arg_name)
                if new_arg_value != origin_arg_value:
                    if var_name not in var_changed_error_massage.keys():
                        var_changed_error_massage[var_name] = {}
                    var_changed_error_massage[var_name][arg_name] = (
                        origin_arg_value, new_arg_value)

    for var_name in vars_name_only_in_origin:
        error, var_error = True, True
        var_deleted_error_massage.append(var_name)

    for var_name in vars_name_only_in_new:
        var_add_massage.append(var_name)
        if not new_vars.get(var_name).get(DISPENSABLE):
            error, var_error = True, True
            var_add_dispensable_massage.append(var_name)

        # if added var is extra, then no need to check.
        if new_vars.get(var_name).get(EXTRA):
            continue

        # if added var is quant, slim needs to review, needs to register.
        if new_vars.get(var_name).get(QUANT):
            error, var_error = True, True
            var_add_quant_message.append(var_name)

        # if added var is def, inference needs to review, needs to register.
        if not new_vars.get(var_name).get(EXTRA) and not new_vars.get(
                var_name).get(QUANT):
            error, var_error = True, True
            var_add_def_message.append(var_name)

    var_diff_message = {}
    if var_add_massage:
        var_diff_message[ADD] = var_add_massage
    if var_add_dispensable_massage:
        var_diff_message[ADD_DISPENSABLE] = var_add_dispensable_massage
    if var_changed_error_massage:
        var_diff_message[CHANGE] = var_changed_error_massage
    if var_deleted_error_massage:
        var_diff_message[DELETE] = var_deleted_error_massage
    if var_add_quant_message:
        var_diff_message[QUANT] = var_add_quant_message
    if var_add_def_message:
        var_diff_message[DEF] = var_add_def_message

    return var_error, var_diff_message


def diff_attr(ori_attrs, new_attrs):
    global error
    attr_error = False

    attr_changed_error_massage = {}
    attr_added_error_massage = []
    attr_added_def_error_massage = []
    attr_deleted_error_massage = []

    attr_added_quant_message = []
    attr_added_define_message = []

    common_attrs = set(ori_attrs.keys()) & set(new_attrs.keys())
    attrs_only_in_origin = set(ori_attrs.keys()) - set(new_attrs.keys())
    attrs_only_in_new = set(new_attrs.keys()) - set(ori_attrs.keys())

    for attr_name in common_attrs:
        if ori_attrs.get(attr_name) == new_attrs.get(attr_name):
            continue
        else:
            error, attr_error = True, True
            for arg_name in ori_attrs.get(attr_name):
                new_arg_value = new_attrs.get(attr_name, {}).get(arg_name)
                origin_arg_value = ori_attrs.get(attr_name, {}).get(arg_name)
                if new_arg_value != origin_arg_value:
                    if attr_name not in attr_changed_error_massage.keys():
                        attr_changed_error_massage[attr_name] = {}
                    attr_changed_error_massage[attr_name][arg_name] = (
                        origin_arg_value, new_arg_value)

    for attr_name in attrs_only_in_origin:
        error, attr_error = True, True
        attr_deleted_error_massage.append(attr_name)

    for attr_name in attrs_only_in_new:
        attr_added_error_massage.append(attr_name)
        if new_attrs.get(attr_name).get(DEFAULT_VALUE) == None:
            error, attr_error = True, True
            attr_added_def_error_massage.append(attr_name)

        # if added attr is quant, slim needs to review, needs to register
        if new_attrs.get(attr_name).get(QUANT):
            error, var_error = True, True
            attr_added_quant_message.append(attr_name)

        # if added attr is def, inference needs to review, needs to register
        if not new_attrs.get(attr_name).get(EXTRA) and not new_attrs.get(
                attr_name).get(QUANT):
            error, var_error = True, True
            attr_added_define_message.append(attr_name)

    attr_diff_message = {}
    if attr_added_error_massage:
        attr_diff_message[ADD] = attr_added_error_massage
    if attr_added_def_error_massage:
        attr_diff_message[ADD_WITH_DEFAULT] = attr_added_def_error_massage
    if attr_changed_error_massage:
        attr_diff_message[CHANGE] = attr_changed_error_massage
    if attr_deleted_error_massage:
        attr_diff_message[DELETE] = attr_deleted_error_massage
    if attr_added_define_message:
        attr_diff_message[DEF] = attr_added_define_message
    if attr_added_quant_message:
        attr_diff_message[QUANT] = attr_added_quant_message

    return attr_error, attr_diff_message


def check_io_registry(io_type, op, diff):
    checker = OpLastCheckpointChecker()
    results = {}
    for update_type in [ADD]:
        for item in diff.get(update_type, []):
            infos = checker.filter_updates(
                op, version_update_map[io_type][update_type], item)
            if not infos:
                if update_type not in results.keys():
                    results[update_type] = []
                # extra not need to register.
                qaunt_ios = diff.get(QUANT, [])
                def_ios = diff.get(DEF, [])
                if item in qaunt_ios or item in def_ios:
                    results[update_type].append((op, item, io_type))

    return results


def check_attr_registry(op, diff, origin_attrs):
    checker = OpLastCheckpointChecker()
    results = {}
    qaunt_attrs = diff.get(QUANT, [])
    def_attrs = diff.get(DEF, [])
    change_attrs = diff.get(CHANGE, {})
    for update_type in [ADD, CHANGE]:
        for item in diff.get(update_type, {}):
            infos = checker.filter_updates(
                op, version_update_map[ATTRS][update_type], item)
            if not infos:
                if update_type == ADD:
                    if update_type not in results.keys():
                        results[update_type] = []
                    # extra not need to register.
                    if item in qaunt_attrs or item in def_attrs:
                        results[update_type].append((op, item))
                elif update_type == CHANGE:
                    if CHANGE not in results.keys():
                        results[update_type] = {}
                    for attr_name, attr_change in change_attrs.items():
                        # extra not need to register.
                        if not origin_attrs.get(attr_name).get(EXTRA):
                            results[update_type][attr_name] = attr_change

    for update_type in [ADD, CHANGE]:
        if update_type in results.keys() and len(results[update_type]) == 0:
            del results[update_type]
    return results


def compare_op_desc(origin_op_desc, new_op_desc):
    origin = json.loads(origin_op_desc)
    new = json.loads(new_op_desc)
    desc_error_message = {}
    version_error_message = {}
    if origin_op_desc == new_op_desc:
        return desc_error_message, version_error_message

    for op_type in origin:
        # no need to compare if the operator is deleted
        if op_type not in new:
            continue

        origin_info = origin.get(op_type, {})
        new_info = new.get(op_type, {})

        origin_inputs = origin_info.get(INPUTS, {})
        new_inputs = new_info.get(INPUTS, {})
        ins_error, ins_diff = diff_vars(origin_inputs, new_inputs)
        ins_version_errors = check_io_registry(INPUTS, op_type, ins_diff)

        origin_outputs = origin_info.get(OUTPUTS, {})
        new_outputs = new_info.get(OUTPUTS, {})
        outs_error, outs_diff = diff_vars(origin_outputs, new_outputs)
        outs_version_errors = check_io_registry(OUTPUTS, op_type, outs_diff)

        origin_attrs = origin_info.get(ATTRS, {})
        new_attrs = new_info.get(ATTRS, {})
        attrs_error, attrs_diff = diff_attr(origin_attrs, new_attrs)
        attrs_version_errors = check_attr_registry(op_type, attrs_diff,
                                                   origin_attrs)

        if ins_diff:
            desc_error_message.setdefault(op_type, {})[INPUTS] = ins_diff
        if outs_diff:
            desc_error_message.setdefault(op_type, {})[OUTPUTS] = outs_diff
        if attrs_diff:
            desc_error_message.setdefault(op_type, {})[ATTRS] = attrs_diff

        if ins_version_errors:
            version_error_message.setdefault(op_type,
                                             {})[INPUTS] = ins_version_errors
        if outs_version_errors:
            version_error_message.setdefault(op_type,
                                             {})[OUTPUTS] = outs_version_errors
        if attrs_version_errors:
            version_error_message.setdefault(op_type,
                                             {})[ATTRS] = attrs_version_errors

    return desc_error_message, version_error_message


def print_desc_error_message(error_message):
    print("\n======================= \n"
          "Op desc error for the changes of Inputs/Outputs/Attrs of OPs:\n")
    for op_name in error_message:
        print("For OP '{}':".format(op_name))

        # 1. print inputs error message
        Inputs_error = error_message.get(op_name, {}).get(INPUTS, {})
        for name in Inputs_error.get(ADD_DISPENSABLE, {}):
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

        for name in Inputs_error.get(QUANT, {}):
            print(" * The added Input '{}' is `quant`, need slim to review.".
                  format(name))

        for name in Inputs_error.get(DEF, {}):
            print(" * The added Input '{}' is `def`, need inference to review.".
                  format(name))

        # 2. print outputs error message
        Outputs_error = error_message.get(op_name, {}).get(OUTPUTS, {})
        for name in Outputs_error.get(ADD_DISPENSABLE, {}):
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

        for name in Outputs_error.get(QUANT, {}):
            print(" * The added Output '{}' is `quant`, need slim to review.".
                  format(name))

        for name in Outputs_error.get(DEF, {}):
            print(
                " * The added Output '{}' is `def`, need inference to review.".
                format(name))

        # 3. print attrs error message
        attrs_error = error_message.get(op_name, {}).get(ATTRS, {})
        for name in attrs_error.get(ADD_WITH_DEFAULT, {}):
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

        for name in attrs_error.get(QUANT, {}):
            # TODO(Wilber):
            print(" * The added attr '{}' is `quant`, need slim to review.".
                  format(name))

        for name in attrs_error.get(DEF, {}):
            # TODO(Wilber):
            print(" * The added attr '{}' is `def`, need inference to review.".
                  format(name))


def print_version_error_message(error_message):
    print(
        "\n======================= \n"
        "Operator registration error for the changes of Inputs/Outputs/Attrs of OPs:\n"
    )
    for op_name in error_message:
        print("For OP '{}':".format(op_name))

        # 1. print inputs error message
        inputs_error = error_message.get(op_name, {}).get(INPUTS, {})
        error_list = inputs_error.get(ADD, [])
        if error_list:
            for tup in error_list:
                print(" * The added input '{}' is not yet registered.".format(
                    tup[1]))

        # 2. print outputs error message
        outputs_error = error_message.get(op_name, {}).get(OUTPUTS, {})
        error_list = outputs_error.get(ADD, [])
        if error_list:
            for tup in error_list:
                print(" * The added output '{}' is not yet registered.".format(
                    tup[1]))

        #3. print attrs error message
        attrs_error = error_message.get(op_name, {}).get(ATTRS, {})
        error_list = attrs_error.get(ADD, [])
        if error_list:
            for tup in error_list:
                print(" * The added attribute '{}' is not yet registered.".
                      format(tup[1]))
        error_dic = error_message.get(op_name, {}).get(ATTRS, {}).get(CHANGE,
                                                                      {})
        for key, val in error_dic.items():
            print(" * The change of attribute '{}' is not yet registered.".
                  format(key))


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

    desc_error_message, version_error_message = compare_op_desc(origin_op_desc,
                                                                new_op_desc)
    if error:
        print("-" * 30)
        print_desc_error_message(desc_error_message)
        print_version_error_message(version_error_message)
        print("-" * 30)
else:
    print("Usage: python check_op_desc.py OP_DESC_DEV.spec OP_DESC_PR.spec")
