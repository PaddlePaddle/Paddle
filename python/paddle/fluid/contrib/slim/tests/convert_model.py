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
        '--model_path', type=str, default='', help='A path to a model.')
    parser.add_argument(
        '--model_save_path',
        type=str,
        default='',
        help='A path to the converted model.')
    parser.add_argument(
        '--file_name',
        type=str,
        default='',
        help='A name to save file. Default - name from model path will be used')
    parser.add_argument(
        '--pdf',
        type=bool,
        default=False,
        help='Convert dot file to pdf. Default False.')

    test_args, args = parser.parse_known_args(namespace=unittest)
    return test_args, sys.argv[:1] + args


def convert_model(original_path, save_path, file_name, to_pdf):
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
        graph = IrGraph(core.Graph(inference_program.desc), for_test=True)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model_name = os.path.basename(os.path.normpath(save_path))
        if file_name is '':
            file_name = model_name
        graph.draw(
            save_path, file_name, graph.all_op_nodes(), convert_to_pdf=to_pdf)
        print("Success! Generated"),
        print("dot file" if to_pdf is False else "dot and pdf files"),
        print("for {0} model, that can be found at {1} named {2}.\n".format(
            model_name, save_path, file_name))


if __name__ == '__main__':
    global test_args
    test_args, remaining_args = parse_args()
    convert_model(test_args.model_path, test_args.model_save_path,
                  test_args.file_name, test_args.pdf)
