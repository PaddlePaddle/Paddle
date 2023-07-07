# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
"""
This script simply removes grad kernels. You should use this script
when cmake ON_INFER=ON, which can greatly reduce the volume of the inference library.
"""

import glob
import os


def is_balanced(content):
    """
    Check whether sequence contains valid parenthesis.
    Args:
       content (str): content of string.

    Returns:
        boolean: True if sequence contains valid parenthesis.
    """

    if content.find('{') == -1:
        return False
    stack = []
    push_chars, pop_chars = '({', ')}'
    for c in content:
        if c in push_chars:
            stack.append(c)
        elif c in pop_chars:
            if not len(stack):
                return False
            else:
                stack_top = stack.pop()
                balancing_bracket = push_chars[pop_chars.index(c)]
                if stack_top != balancing_bracket:
                    return False
    return not stack


def grad_kernel_definition(content, kernel_pattern, grad_pattern):
    """
    Args:
       content(str): file content
       kernel_pattern(str): kernel pattern
       grad_pattern(str): grad pattern

    Returns:
        (list, int): grad kernel definitions in file and count.
    """

    results = []
    count = 0
    start = 0
    lens = len(content)
    while True:
        index = content.find(kernel_pattern, start)
        if index == -1:
            return results, count
        i = index + 1
        while i <= lens:
            check_str = content[index:i]
            if is_balanced(check_str):
                if check_str.find(grad_pattern) != -1:
                    results.append(check_str)
                    count += 1
                start = i
                break
            i += 1
        else:
            return results, count


def remove_grad_kernels(dry_run=False):
    """
    Args:
       dry_run(bool): whether just print

    Returns:
        int: number of kernel(grad) removed
    """

    pd_kernel_pattern = 'PD_REGISTER_STRUCT_KERNEL'
    register_op_pd_kernel_count = 0
    matches = []

    tool_dir = os.path.dirname(os.path.abspath(__file__))
    all_op = glob.glob(
        os.path.join(tool_dir, '../paddle/fluid/operators/**/*.cc'),
        recursive=True,
    )
    all_op += glob.glob(
        os.path.join(tool_dir, '../paddle/fluid/operators/**/*.cu'),
        recursive=True,
    )

    for op_file in all_op:
        with open(op_file, 'r', encoding='utf-8') as f:
            content = ''.join(f.readlines())

            pd_kernel, pd_kernel_count = grad_kernel_definition(
                content, pd_kernel_pattern, '_grad,'
            )

            register_op_pd_kernel_count += pd_kernel_count

            matches.extend(pd_kernel)

        for to_remove in matches:
            content = content.replace(to_remove, '')
            if dry_run:
                print(op_file, to_remove)

        if not dry_run:
            with open(op_file, 'w', encoding='utf-8') as f:
                f.write(content)

    return register_op_pd_kernel_count
