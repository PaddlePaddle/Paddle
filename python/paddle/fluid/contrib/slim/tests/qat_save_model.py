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
        '--save_model_path',
        type=str,
        default='',
        help='Saved transformed model to the path')

    test_args, args = parser.parse_known_args(namespace=unittest)
    return test_args, sys.argv[:1] + args


def save_transformed_model(args):
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    inference_scope = fluid.executor.global_scope()
    with fluid.scope_guard(inference_scope):
        if os.path.exists(os.path.join(args.qat_model_path, '__model__')):
            [inference_program, feed_target_names, fetch_targets
             ] = fluid.io.load_inference_model(args.qat_model_path, exe)
        else:
            [inference_program, feed_target_names,
             fetch_targets] = fluid.io.load_inference_model(
                 args.qat_model_path, exe, 'model', 'params')

        graph = IrGraph(core.Graph(inference_program.desc), for_test=True)
        transform_to_mkldnn_int8_pass = FakeQAT2MkldnnINT8PerfPass(
            _scope=inference_scope, _place=place, _core=core)
        graph = transform_to_mkldnn_int8_pass.apply(graph)

        inference_program = graph.to_program()
        if args.save_model_path:
            with fluid.scope_guard(inference_scope):
                fluid.io.save_inference_model(args.save_model_path,
                                              feed_target_names, fetch_targets,
                                              exe, inference_program)


if __name__ == '__main__':
    global test_case_args
    test_case_args, remaining_args = parse_args()
    save_transformed_model(test_case_args)
