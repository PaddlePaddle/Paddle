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

import os
import sys


def group_case_for_parallel(rootPath):
    """group cases"""

    # wget file
    for filename in [
        'nightly_case',
        'single_card_tests',
        'single_card_tests_mem0',
        'multiple_card_tests',
        'multiple_card_tests_mem0',
        'exclusive_card_tests',
        'exclusive_card_tests_mem0',
    ]:
        OS_NAME = sys.platform
        if OS_NAME.startswith('win'):
            os.system(
                f'cd {rootPath}/tools && wget --no-proxy https://paddle-windows.bj.bcebos.com/pre_test_bak_20230908/{filename} --no-check-certificate'
            )
        else:
            os.system(
                f'cd {rootPath}/tools && wget --no-proxy https://paddle-docker-tar.bj.bcebos.com/pre_test_bak_20230908/{filename} --no-check-certificate'
            )

    # get nightly tests
    nightly_tests_file = open(f'{rootPath}/tools/nightly_case', 'r')
    nightly_tests = nightly_tests_file.read().strip().split('\n')
    nightly_tests_file.close()

    parallel_case_file_list = [
        f'{rootPath}/tools/single_card_tests_mem0',
        f'{rootPath}/tools/single_card_tests',
        f'{rootPath}/tools/multiple_card_tests_mem0',
        f'{rootPath}/tools/multiple_card_tests',
        f'{rootPath}/tools/exclusive_card_tests_mem0',
        f'{rootPath}/tools/exclusive_card_tests',
    ]
    case_file = f'{rootPath}/build/ut_list'
    if os.path.exists(case_file):
        f = open(case_file, 'r')
        all_need_run_cases = f.read().strip().split('\n')
        if len(all_need_run_cases) == 1 and all_need_run_cases[0] == '':
            f.close()
            case_file = f'{rootPath}/build/all_ut_list'
            f = open(case_file, 'r')
            all_need_run_cases = f.read().strip().split('\n')
    else:
        case_file = f'{rootPath}/build/all_ut_list'
        f = open(case_file, 'r')
        all_need_run_cases = f.read().strip().split('\n')

    print(f"case_file: {case_file}")

    all_group_case = []
    cinn_case_file_list = ['pr_ci_cinn_ut_list', 'pr_ci_cinn_gpu_ut_list']
    for filename in cinn_case_file_list:
        if not os.path.exists(f'{rootPath}/build/{filename}'):
            continue
        cinn_tests_file = open(f'{rootPath}/build/{filename}', 'r')
        cinn_cases = cinn_tests_file.read().strip().split('\n')
        cinn_tests_file.close()
        new_cinn_cases_list = list(
            set(all_need_run_cases).intersection(set(cinn_cases))
        )
        if len(new_cinn_cases_list) != 0:
            all_group_case += new_cinn_cases_list
            all_need_run_cases = list(
                set(all_need_run_cases).difference(set(all_group_case))
            )
        cases = '$|^'.join(new_cinn_cases_list)
        cases = f'^job$|^{cases}$'
        cinn_f = open(f'{rootPath}/tools/new_{filename}', 'w')
        cinn_f.write(cases + '\n')
        cinn_f.close()
        print(
            f"new_{filename} is same as {filename}: ",
            len(new_cinn_cases_list) == len(cinn_cases),
            f"case num of new_{filename}is: ",
            len(new_cinn_cases_list),
        )

    for filename in parallel_case_file_list:
        fi = open(filename, 'r')
        new_f = open(f'{filename}_new', 'w')
        lines = fi.readlines()
        new_case_file_list = []
        for line in lines:
            case_line_list = line.replace('^', '').replace('|', '').split('$')
            new_case_line_list = list(
                set(all_need_run_cases).intersection(set(case_line_list))
            )
            if len(new_case_line_list) != 0:
                new_case_file_list.append(new_case_line_list)
                all_group_case += new_case_line_list
                all_need_run_cases = list(
                    set(all_need_run_cases).difference(set(all_group_case))
                )

        for line in new_case_file_list:
            cases = '$|^'.join(case for case in line)
            cases = f'^job$|^{cases}$'
            new_f.write(cases + '\n')
        fi.close()
        new_f.close()

    # no parallel cases
    max_case_num_per_line = 100
    no_parallel_case_line_list = []
    cases = '^job'
    if len(all_need_run_cases) != 0:
        cnt = 0
        for case in all_need_run_cases:
            if case not in nightly_tests:
                cases = cases + f'$|^{case}'
                cnt += 1
                if cnt == max_case_num_per_line:
                    cases = f'{cases}$'
                    no_parallel_case_line_list.append(cases)
                    cnt = 0
                    cases = '^job'

        cases = f'{cases}$'
        no_parallel_case_line_list.append(cases)

    new_f = open(f'{rootPath}/tools/no_parallel_case_file', 'w')
    for case_line in no_parallel_case_line_list:
        new_f.write(case_line + '\n')
    new_f.close()
    f.close()


if __name__ == "__main__":
    rootPath = sys.argv[1]
    group_case_for_parallel(rootPath)
