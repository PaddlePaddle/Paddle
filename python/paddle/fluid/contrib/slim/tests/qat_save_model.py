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
from paddle.fluid.contrib.slim.quantization import FakeQAT2MkldnnINT8PerfPass
from paddle.fluid import core


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--qat_model_path', type=str, default='', help='A path to a QAT model.')
    parser.add_argument(
        '--qat_fp32_model_path',
        type=str,
        default='',
        help='Saved fused fp32 model')
    parser.add_argument(
        '--qat_int8_model_path',
        type=str,
        default='',
        help='Saved fused and quantized INT8 model')

    test_args, args = parser.parse_known_args(namespace=unittest)
    return test_args, sys.argv[:1] + args


def save_transformed_model(saved_type, original_path, saved_path):
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

        transform_to_mkldnn_int8_pass = FakeQAT2MkldnnINT8PerfPass(
            _scope=inference_scope, _place=place, _core=core)

        graph = IrGraph(core.Graph(inference_program.desc), for_test=True)
        if saved_type == 'FP32':
            graph = transform_to_mkldnn_int8_pass.apply_fp32_passes(graph)
        else:
            graph = transform_to_mkldnn_int8_pass.apply(graph)
        inference_program = graph.to_program()
        with fluid.scope_guard(inference_scope):
            fluid.io.save_inference_model(saved_path, feed_target_names,
                                          fetch_targets, exe, inference_program)
        print("Success! QAT_{0} model can be found at {1}\n".format(saved_type,
                                                                    saved_path))


if __name__ == '__main__':
    global test_args
    test_args, remaining_args = parse_args()
    if test_args.qat_fp32_model_path:
        save_transformed_model('FP32', test_args.qat_model_path,
                               test_args.qat_fp32_model_path)
    if test_args.qat_int8_model_path:
        save_transformed_model('INT8', test_args.qat_model_path,
                               test_args.qat_int8_model_path)
