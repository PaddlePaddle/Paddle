#   copyright (c) 2019 paddlepaddle authors. all rights reserved.
#
# licensed under the apache license, version 2.0 (the "license");
# you may not use this file except in compliance with the license.
# you may obtain a copy of the license at
#
#     http://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "as is" basis,
# without warranties or conditions of any kind, either express or implied.
# see the license for the specific language governing permissions and
# limitations under the license.

import unittest
import os
import sys
import argparse
import logging
import struct
import six
import numpy as np
import time
import paddle
import paddle.fluid as fluid
from paddle.fluid.framework import IrGraph
from paddle.fluid.contrib.slim.quantization import Qat2Int8MkldnnPass
from paddle.fluid import core


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--qat_model_path', type=str, default='', help='A path to a QAT model.')
    parser.add_argument(
        '--fp32_model_save_path',
        type=str,
        default='',
        help='Saved optimized fp32 model')
    parser.add_argument(
        '--int8_model_save_path',
        type=str,
        default='',
        help='Saved optimized and quantized INT8 model')
    parser.add_argument(
        '--ops_to_quantize',
        type=str,
        default='',
        help='A comma separated list of operators to quantize. Only quantizable operators are taken into account. If the option is not used, an attempt to quantize all quantizable operators will be made.'
    )

    test_args, args = parser.parse_known_args(namespace=unittest)
    return test_args, sys.argv[:1] + args


def transform_and_save_model(original_path, save_path, save_type):
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    inference_scope = fluid.executor.global_scope()
    with fluid.scope_guard(inference_scope):
        if os.path.exists(os.path.join(original_path, '__model__')):
            [inference_program, feed_target_names,
             fetch_targets] = fluid.io.load_inference_model(original_path, exe)
        else:
            [inference_program, feed_target_names,
             fetch_targets] = fluid.io.load_inference_model(original_path, exe,
                                                            'model', 'params')

        ops_to_quantize = set()
        if len(test_args.ops_to_quantize) > 0:
            ops_to_quantize = set(test_args.ops_to_quantize.split(','))

        transform_to_mkldnn_int8_pass = Qat2Int8MkldnnPass(
            ops_to_quantize, _scope=inference_scope, _place=place, _core=core)

        graph = IrGraph(core.Graph(inference_program.desc), for_test=True)
        if save_type == 'FP32':
            graph = transform_to_mkldnn_int8_pass.apply_fp32(graph)
        elif save_type == 'INT8':
            graph = transform_to_mkldnn_int8_pass.apply(graph)
        inference_program = graph.to_program()
        with fluid.scope_guard(inference_scope):
            fluid.io.save_inference_model(save_path, feed_target_names,
                                          fetch_targets, exe, inference_program)
        print("Success! Transformed QAT_{0} model can be found at {1}\n".format(
            save_type, save_path))


if __name__ == '__main__':
    global test_args
    test_args, remaining_args = parse_args()
    if test_args.fp32_model_save_path:
        transform_and_save_model(test_args.qat_model_path,
                                 test_args.fp32_model_save_path, 'FP32')
    if test_args.int8_model_save_path:
        transform_and_save_model(test_args.qat_model_path,
                                 test_args.int8_model_save_path, 'INT8')
