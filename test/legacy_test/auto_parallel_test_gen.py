# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

IMPORT_PACKAGE_TEMPLATE = """

import pathlib
import pickle
import sys
"""

IMPORT_FORWARD_TEST_CLASS_TEMPLATE = """

sys.path.append(
    str(pathlib.Path(__file__).resolve().parents[0] / 'test/legacy_test')
)
from auto_parallel_op_test import AutoParallelForwardChecker
"""

IMPORT_GRAD_TEST_CLASS_TEMPLATE = """

sys.path.append(
    str(pathlib.Path(__file__).resolve().parents[0] / 'test/legacy_test')
)
from auto_parallel_op_test import AutoParallelGradChecker
"""

LOAD_TEST_INFO_TEMPLATE = """

file = open(
    "{test_info_path}", "rb"
)
test_info = pickle.load(file)
file.close()
"""

FORWARD_TEST_BODY_TEMPLATE = """

auto_parallel_forward_checker = AutoParallelForwardChecker(
    test_info["op_type"],
    test_info["python_api"],
    test_info["dtype"],
    test_info["input_specs"],
    test_info["inputs"],
    test_info["attrs"],
    test_info["outputs"],
    test_info["place"],
    test_info["python_out_sig"],
)
auto_parallel_forward_checker.check()
"""

GRAD_TEST_BODY_TEMPLATE = """

auto_parallel_forward_checker = AutoParallelGradChecker(
    test_info["op_type"],
    test_info["python_api"],
    test_info["dtype"],
    test_info["input_specs"],
    test_info["inputs"],
    test_info["attrs"],
    test_info["outputs"],
    test_info["place"],
    test_info["inputs_to_check"],
    test_info["output_names"],
    test_info["user_defined_grad_outputs"],
    test_info["python_out_sig"],
)
auto_parallel_forward_checker.check()
"""


def gen_auto_parallel_test_file(check_grad, test_info_path, test_file_path):
    test_code = ''
    test_code += IMPORT_PACKAGE_TEMPLATE
    test_code += (
        IMPORT_GRAD_TEST_CLASS_TEMPLATE
        if check_grad
        else IMPORT_FORWARD_TEST_CLASS_TEMPLATE
    )
    test_code += LOAD_TEST_INFO_TEMPLATE.format(test_info_path=test_info_path)
    test_code += (
        GRAD_TEST_BODY_TEMPLATE if check_grad else FORWARD_TEST_BODY_TEMPLATE
    )
    with open(test_file_path, "w") as f:
        f.write(test_code)


def main():
    parser = argparse.ArgumentParser(
        description='Generate auto_parallel test code for op'
    )
    parser.add_argument(
        '--gen_test_file_path',
        type=str,
        help='path to generated test file',
    )

    parser.add_argument(
        '--test_info_path',
        type=str,
        help='path to load test info',
    )

    parser.add_argument(
        '--check_grad',
        type=bool,
        default=False,
        help='determine whether the test is forward or grad',
    )

    options = parser.parse_args()

    test_file_path = options.gen_test_file_path
    test_info_path = options.test_info_path
    check_grad = options.check_grad

    gen_auto_parallel_test_file(check_grad, test_info_path, test_file_path)


if __name__ == '__main__':
    main()
