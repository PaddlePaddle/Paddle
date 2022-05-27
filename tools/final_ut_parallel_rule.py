# -*- coding: utf-8 -*-

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
import time
import json
import datetime
import codecs
import sys


def classify_cases_by_mem(rootPath):
    """classify cases by mem"""
    case_filename = '%s/build/classify_case_by_cardNum.txt' % rootPath
    always_timeout_list = [
        "test_post_training_quantization_mnist",
        "test_post_training_quantization_while", "test_mkldnn_log_softmax_op",
        "test_mkldnn_matmulv2_op", "test_mkldnn_shape_op",
        "interceptor_pipeline_short_path_test",
        "interceptor_pipeline_long_path_test", "test_cpuonly_spawn",
        "test_quant2_int8_resnet50_channelwise_mkldnn",
        "test_quant2_int8_resnet50_mkldnn",
        "test_quant2_int8_resnet50_range_mkldnn"
    ]
    f = open(case_filename)
    lines = f.readlines()
    all_tests_by_card = {}
    for line in lines:
        if line.startswith('single_card_tests:'):
            all_tests_by_card['single_card_tests'] = []
            line = line.split('single_card_tests: ^job$|')[1].split('|')
            for case in line:
                case = case.replace('^', '').replace('$', '').strip()
                all_tests_by_card['single_card_tests'].append(case)
        elif line.startswith('multiple_card_tests:'):
            all_tests_by_card['multiple_card_tests'] = []
            line = line.split('multiple_card_tests: ^job$|')[1].split('|')
            for case in line:
                case = case.replace('^', '').replace('$', '').strip()
                all_tests_by_card['multiple_card_tests'].append(case)
        elif line.startswith('exclusive_card_tests:'):
            all_tests_by_card['exclusive_card_tests'] = []
            line = line.split('exclusive_card_tests: ^job$')[1].split('|')
            for case in line:
                case = case.replace('^', '').replace('$', '').strip()
                all_tests_by_card['exclusive_card_tests'].append(case)

    with open("/pre_test/ut_mem_map.json", 'r') as load_f:
        new_lastest_mem = json.load(load_f)

    no_parallel_case = '^job$'
    for cardType in all_tests_by_card:
        case_mem_0 = '^job$'
        case_mem_1 = {}
        for case in all_tests_by_card[cardType]:
            if case in always_timeout_list:
                no_parallel_case = no_parallel_case + '|^' + case + '$'
                continue
            if case not in new_lastest_mem:
                no_parallel_case = no_parallel_case + '|^' + case + '$'
                continue
            #mem = 0
            if new_lastest_mem[case]["mem_nvidia"] == 0:
                case_mem_0 = case_mem_0 + '|^' + case + '$'
            #mem != 0
            else:
                case_mem_1[case] = new_lastest_mem[case]["mem_nvidia"]

        with open('/pre_test/%s_mem0' % cardType, 'w') as f:
            f.write(case_mem_0)
            f.close()

        case_mem_1_sort = sorted(case_mem_1.items(), key=lambda x: x[1])
        case_mem_1_line = '^job$'
        mem_1_sum = 0
        with open('/pre_test/%s' % cardType, 'w') as f:
            for index in case_mem_1_sort:
                if mem_1_sum < 14 * 1024 * 2:
                    mem_1_sum += index[1]
                    case_mem_1_line = case_mem_1_line + '|^' + index[0] + '$'
                else:
                    f.write(case_mem_1_line + '\n')
                    case_mem_1_line = '^job$|^' + case + '$'
                    mem_1_sum = index[1]
            f.write(case_mem_1_line + '\n')
        f.close()

    with open('/pre_test/no_parallel_case', 'w') as f:
        f.write(no_parallel_case + '\n')

    os.system('mv %s/build/nightly_case /pre_test/' % rootPath)


if __name__ == '__main__':
    rootPath = sys.argv[1]
    classify_cases_by_mem(rootPath)
