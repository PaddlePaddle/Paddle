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

    #wget file
    for filename in [
            'nightly_case', 'single_card_tests', 'single_card_tests_mem0',
            'multiple_card_tests', 'multiple_card_tests_mem0',
            'exclusive_card_tests', 'exclusive_card_tests_mem0'
    ]:
        os.system(
            'cd %s/tools && wget --no-proxy https://paddle-docker-tar.bj.bcebos.com/pre_test_bak/%s --no-check-certificate'
            % (rootPath, filename))

    #get nightly tests
    nightly_tests_file = open('%s/tools/nightly_case' % rootPath, 'r')
    nightly_tests = nightly_tests_file.read().strip().split('\n')
    nightly_tests_file.close()

    parallel_case_file_list = [
        '%s/tools/single_card_tests_mem0' % rootPath,
        '%s/tools/single_card_tests' % rootPath,
        '%s/tools/multiple_card_tests_mem0' % rootPath,
        '%s/tools/multiple_card_tests' % rootPath,
        '%s/tools/exclusive_card_tests_mem0' % rootPath,
        '%s/tools/exclusive_card_tests' % rootPath
    ]
    case_file = '%s/build/ut_list' % rootPath
    if os.path.exists(case_file):
        f = open(case_file, 'r')
        all_need_run_cases = f.read().strip().split('\n')
        if len(all_need_run_cases) == 1 and all_need_run_cases[0] == '':
            f.close()
            case_file = '%s/build/all_ut_list' % rootPath
            f = open(case_file, 'r')
            all_need_run_cases = f.read().strip().split('\n')
    else:
        case_file = '%s/build/all_ut_list' % rootPath
        f = open(case_file, 'r')
        all_need_run_cases = f.read().strip().split('\n')

    print("case_file: %s" % case_file)

    all_group_case = []
    for filename in parallel_case_file_list:
        fi = open(filename, 'r')
        new_f = open('%s_new' % filename, 'w')
        lines = fi.readlines()
        new_case_file_list = []
        for line in lines:
            case_line_list = line.replace('^', '').replace('|', '').split('$')
            new_case_line_list = list(
                set(all_need_run_cases).intersection(set(case_line_list)))
            if len(new_case_line_list) != 0:
                new_case_file_list.append(new_case_line_list)
                all_group_case += new_case_line_list
                all_need_run_cases = list(
                    set(all_need_run_cases).difference(set(all_group_case)))

        for line in new_case_file_list:
            cases = '$|^'.join(case for case in line)
            cases = '^job$|^%s$' % cases
            new_f.write(cases + '\n')
        fi.close()
        new_f.close()

    #no parallel cases
    cases = '^job'
    if len(all_need_run_cases) != 0:
        for case in all_need_run_cases:
            if case not in nightly_tests:
                cases = cases + '$|^%s' % case
        cases = '%s$' % cases

    new_f = open('%s/tools/no_parallel_case_file' % rootPath, 'w')
    new_f.write(cases + '\n')
    new_f.close()
    f.close()


if __name__ == "__main__":
    rootPath = sys.argv[1]
    group_case_for_parallel(rootPath)
