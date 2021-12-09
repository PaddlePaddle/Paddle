# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""To get a list of prec ut """

import sys
import os
import platform


def get_prec_ut_list(all_test_cases, prec_test_cases):
    """Select the ut that needs to be executed"""
    all_test_cases_list = all_test_cases.strip().split("\n")
    prec_test_cases_list = prec_test_cases.strip().split("\n")
    all_test_cases_list_new = [item.rstrip() for item in all_test_cases_list]
    prec_test_cases_list_new = [item.rstrip() for item in prec_test_cases_list]

    if len(prec_test_cases) == 0:
        return

    case_to_run = ['test_prec_ut']
    for case in all_test_cases_list_new:
        if case in prec_test_cases_list_new:
            case_to_run.append(case)
        else:
            print("{} will not run in PRECISION_TEST mode.".format(case))

    with open(file_path, 'w') as f:
        f.write('\n'.join(case_to_run))


if __name__ == '__main__':
    # get prec cases lists
    with open('ut_list', 'r') as f:
        prec_test_cases = f.read()

    # sys.argv[1] may exceed max_arg_length when busybox run parallel_UT_rule in windows
    BUILD_DIR = os.getcwd()
    file_path = os.path.join(BUILD_DIR, 'all_ut_list')
    with open(file_path, 'r') as f:
        all_test_cases = f.read()
    #prec_test_cases = sys.argv[2]
    get_prec_ut_list(all_test_cases, prec_test_cases)
