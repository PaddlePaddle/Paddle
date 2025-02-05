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

import argparse
import math
import os
import re
from threading import Thread

# some particular ops
black_list = [
    # special op
    'test_custom_relu_op_setup',
    'test_custom_relu_op_jit',
    'test_python_operator_overriding',
    'test_c_embedding_op',
    # train op
    'test_imperative_optimizer',
    'test_imperative_optimizer_v2',
    'test_momentum_op',
    'test_sgd_op',
    'test_sgd_op_bf16',
    'test_warpctc_op',
    # case too large
    'test_reduce_op',
    'test_transpose_op',
]

op_diff_list = [
    # diff<1E-7,it's right
    'test_elementwise_mul_op'
]


def parse_arguments():
    """
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--shell_name',
        type=str,
        default='get_op_list.sh',
        help='please input right name',
    )
    parser.add_argument(
        '--op_list_file',
        type=str,
        default='list_op.txt',
        help='please input right name',
    )
    return parser.parse_args()


def search_file(file_name, path, file_path):
    """
    :param file_name:target
    :param path: to search this path
    :param file_path: result
    :return:
    """
    for item in os.listdir(path):
        if os.path.isdir(os.path.join(path, item)):
            search_file(file_name, os.path.join(path, item), file_path)
        else:
            if file_name in item:
                file_path.append(os.path.join(path, file_name))


def get_prefix(line, end_char='d'):
    """
    :param line: string_demo
    :param end_char: copy the prefix of string_demo until end_char
    :return: prefix
    """
    i = 0
    prefix = ''
    while line[i] != end_char:
        prefix += line[i]
        i += 1
    return prefix


def add_import_skip_return(file, pattern_import, pattern_skip, pattern_return):
    """
    :param file: the file need to be changed
    :param pattern_import: import skip
    :param pattern_skip: @skip
    :param pattern_return: add return
    :return:
    """
    pattern_1 = re.compile(pattern_import)
    pattern_2 = re.compile(pattern_skip)
    pattern_3 = re.compile(pattern_return)
    file_data = ""

    # change file
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            # add import skip_check_grad_ci
            match_obj = pattern_1.search(line)
            if match_obj is not None:
                line = line[:-1] + ", skip_check_grad_ci\n"
                print("### add import skip_check_grad_ci ####")

            # add @skip_check_grad_ci
            match_obj = pattern_2.search(line)
            if match_obj is not None:
                file_data += (
                    "@skip_check_grad_ci(reason='jetson do n0t need this !')\n"
                )
                print("### add @skip_check_grad_ci ####")

            # delete test_grad_output
            match_obj = pattern_3.search(line)
            if match_obj is not None:
                file_data += line
                file_data += get_prefix(line)
                file_data += "    return\n"
                print("### add return for function ####")
                continue
            file_data += line
    # save file
    with open(file, "w", encoding="utf-8") as f:
        f.write(file_data)


def get_op_list(op_list_file='list_op.txt'):
    """
    :param op_list_file: op list file
    :return: list of op
    """
    op_list = []
    with open(op_list_file, "r", encoding="utf-8") as f:
        for line in f:
            if line in black_list:
                continue
            # delete /n
            op_list.append(line[:-1])
    return op_list


def set_diff_value(file, atol="1e-5", inplace_atol="1e-7"):
    """
    :param file: refer to op_test.py
    :param atol: refer to op_test.py
    :param inplace_atol:
    :return:
    """
    os.system(
        r"sed -i 's/self.check_output(/self\.check_output\(atol="
        + atol
        + ",inplace_atol="
        + inplace_atol
        + ",/g' "
        + file
    )


def change_op_file(start=0, end=0, op_list_file='list_op.txt', path='.'):
    """
    :param start:
    :param end:
    :param op_list_file: op_list
    :param path: just the file in this path
    :return:
    """
    test_op_list = get_op_list(op_list_file)
    file_path = []
    for id in range(start, end):
        item = test_op_list[id]
        print(id, ":", item)
        search_file(item + '.py', os.path.abspath(path), file_path)
        if len(file_path) == 0:
            print("'", item, "' is not a python file!")
            continue
        file_with_path = file_path[0]
        # pattern
        pattern_import = ".*import OpTest.*"
        pattern_skip = r"^class .*\(OpTest\):$"
        pattern_return = r"def test.*grad.*\):$"
        # change file
        add_import_skip_return(
            file_with_path, pattern_import, pattern_skip, pattern_return
        )
        # op_diff
        if item in op_diff_list:
            set_diff_value(file_with_path)

        file_path.clear()


def run_multi_thread(list_file, thread_num=4):
    """
    :param list_file:
    :param thread_num:
    :return:
    """
    length = len(get_op_list(list_file))
    thread_list = []
    start = 0
    end = 0
    for item in range(thread_num):
        # set argument
        start = math.floor(item / thread_num * length)
        end = math.floor((item + 1) / thread_num * length)
        print("thread num-", item, ":", start, end)
        thread = Thread(target=change_op_file, args=(start, end))
        thread_list.append(thread)
        # start a thread
        thread.start()

    # wait all thread
    for item in thread_list:
        item.join()

    # add a flag
    with open("flag_change_file.txt", "w", encoding="utf-8") as f:
        f.write("change successfully!")

    print("------change successfully!-------")


def transform_list_to_str(list_op):
    """
    :param list_op:
    :return:
    """
    res = ""
    for item in list_op:
        tmp = "^" + item + "$|"
        res += tmp
    return res[:-1]


def run_file_change(op_list_file):
    """
    if file has changed, the file should not be changed again.
    :param op_list_file:
    :return:
    """
    if os.path.exists("flag_change_file.txt"):
        print(
            "-----maybe op_file has changed, so don't need to change again------"
        )
    else:
        run_multi_thread(op_list_file)


def run_test_first(op_list_file):
    """
    run all op test.
    :param op_list_file:
    :return:
    """
    old_list = get_op_list(op_list_file)
    new_list = filter(lambda x: x not in black_list, old_list)
    op_test = transform_list_to_str(new_list)
    os.system(f'ctest -R "({op_test})" >& test_op_log.txt')


def run_test_second():
    """
    run failed op again.
    :return:
    """
    os.system(
        "sed -n '/(Failed)$/p'  test_op_log.txt | awk '{print $3}' >& rerun_op.txt"
    )
    rerun_list = get_op_list('rerun_op.txt')
    if len(rerun_list):
        print(
            "-------there are "
            + str(len(rerun_list))
            + " op(s) need to rerun!!!-------"
        )
        for failed_op in rerun_list:
            os.system(f'ctest -R "({failed_op})" ')
    else:
        print("-------all op passed successfully!!!-------")


if __name__ == '__main__':
    arg = parse_arguments()
    print("------start get op list!------")
    os.system("bash " + arg.shell_name + " " + arg.op_list_file)
    print("------start change op file!------")
    run_file_change(arg.op_list_file)
    print("------start run  op  test first!------")
    run_test_first(arg.op_list_file)
    print("------start run  failed_op  test!------")
    run_test_second()
    print("------do well!------")
