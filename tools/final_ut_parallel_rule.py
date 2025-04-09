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

import json
import os
import sys


def classify_cases_by_mem(rootPath):
    """classify cases by mem"""
    case_filename = f'{rootPath}/build/classify_case_by_cardNum.txt'
    case_exec_100 = [
        'test_conv_eltwiseadd_bn_fuse_pass',
        'test_trt_convert_pool2d',
        'test_fc_fuse_pass',
        'test_trt_convert_depthwise_conv2d',
        'test_quant2_int8_resnet50_mkldnn',
        'test_conv_elementwise_add_act_fuse_pass',
        'test_trt_convert_conv2d',
        'test_paddle_save_load',
        'test_logical_op',
        'test_pool2d_op',
        'test_conv3d_transpose_op',
        'test_lstmp_op',
        'test_cross_entropy2_op',
        'test_sgd_op',
        'test_imperative_ptq',
        'test_model',
        'test_custom_relu_op_setup',
        'test_dropout_op',
        'test_concat_op',
    ]  # 木桶原理 70s-100s之间的case

    case_exec_200 = [
        'test_post_training_quantization_mnist',
        'test_trt_dynamic_shape_ernie_fp16_ser_deser',
        'test_trt_dynamic_shape_ernie',
        'test_layer_norm_op',
        'trt_quant_int8_yolov3_r50_test',
        'test_gru_op',
        'test_post_training_quantization_while',
        'test_mkldnn_log_softmax_op',
        'test_mkldnn_matmulv2_op',
        'test_mkldnn_shape_op',
        'interceptor_pipeline_short_path_test',
        'interceptor_pipeline_long_path_test',
        'test_cpuonly_spawn',
    ]  # 木桶原理 110s-200s之间的case 以及容易timeout

    case_always_timeout = [
        'test_quant2_int8_resnet50_channelwise_mkldnn',
        'test_parallel_dygraph_unused_variables_gloo',
        'test_seq2seq',
        'test_pool3d_op',
        'test_trilinear_interp_v2_op',
        'test_dropout_op',
        'test_parallel_dygraph_sync_batch_norm',
        'test_conv3d_op',
        'test_quant2_int8_resnet50_range_mkldnn',
    ]  # always timeout

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

    if not os.path.exists("/pre_test"):
        os.mkdir("/pre_test")

    with open("/pre_test/classify_case_by_cardNum.json", "w") as f:
        json.dump(all_tests_by_card, f)

    with open("/pre_test/ut_mem_map.json", 'r') as load_f:
        new_latest_mem = json.load(load_f)
    no_parallel_case = '^job$'
    for cardType in all_tests_by_card:
        case_mem_0 = '^job$'
        case_mem_1 = {}
        for case in all_tests_by_card[cardType]:
            if case in case_exec_100 or case in case_exec_200:
                continue
            if case in case_always_timeout:
                no_parallel_case = no_parallel_case + '|^' + case + '$'
                continue

            if case not in new_latest_mem:
                continue

            # mem = 0
            if new_latest_mem[case]["mem_nvidia"] == 0:
                case_mem_0 = case_mem_0 + '|^' + case + '$'
            # mem != 0
            else:
                case_mem_1[case] = new_latest_mem[case]["mem_nvidia"]

        with open(f'/pre_test/{cardType}_mem0', 'w') as f:
            f.write(case_mem_0)
            f.close()

        case_mem_1_sort = sorted(case_mem_1.items(), key=lambda x: x[1])
        case_mem_1_line = '^job$'
        mem_1_sum = 0
        with open(f'/pre_test/{cardType}', 'w') as f_not_0:
            for index in case_mem_1_sort:
                if mem_1_sum < 14 * 1024 * 2:
                    mem_1_sum += index[1]
                    case_mem_1_line = case_mem_1_line + '|^' + index[0] + '$'
                else:
                    f_not_0.write(case_mem_1_line + '\n')
                    case_mem_1_line = '^job$|^' + index[0] + '$'
                    mem_1_sum = index[1]
            f_not_0.write(case_mem_1_line + '\n')

            if cardType == 'single_card_tests':
                for cases in [case_exec_100, case_exec_200]:
                    case_mem_1_line = '^job$'
                    for case in cases:
                        case_mem_1_line = case_mem_1_line + '|^' + case + '$'
                    f_not_0.write(case_mem_1_line + '\n')
            f_not_0.close()

    os.system(f'cp {rootPath}/build/nightly_case /pre_test/')


if __name__ == '__main__':
    rootPath = sys.argv[1]
    classify_cases_by_mem(rootPath)
