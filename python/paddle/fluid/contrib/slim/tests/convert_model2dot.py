#   copyright (c) 2020 paddlepaddle authors. all rights reserved.
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
import paddle.fluid as fluid
from paddle.fluid.framework import IrGraph
from paddle.fluid import core


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path', type=str, default='', help='A path to a model.')
    parser.add_argument(
        '--save_graph_dir',
        type=str,
        default='',
        help='A path to save the graph.')
    parser.add_argument(
        '--save_graph_name',
        type=str,
        default='',
        help='A name to save the graph. Default - name from model path will be used'
    )

    test_args, args = parser.parse_known_args(namespace=unittest)
    return test_args, sys.argv[:1] + args


def generate_dot_for_model(model_path, save_graph_dir, save_graph_name):
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    inference_scope = fluid.executor.global_scope()
    with fluid.scope_guard(inference_scope):
        if os.path.exists(os.path.join(model_path, '__model__')):
            [inference_program, feed_target_names,
             fetch_targets] = fluid.io.load_inference_model(model_path, exe)
        else:
            [inference_program, feed_target_names,
             fetch_targets] = fluid.io.load_inference_model(model_path, exe,
                                                            'model', 'params')
        graph = IrGraph(core.Graph(inference_program.desc), for_test=True)
        if not os.path.exists(save_graph_dir):
            os.makedirs(save_graph_dir)
        model_name = os.path.basename(os.path.normpath(save_graph_dir))
        if save_graph_name is '':
            save_graph_name = model_name
        graph.draw(save_graph_dir, save_graph_name, graph.all_op_nodes())
        print(
            "Success! Generated dot and pdf files for {0} model, that can be found at {1} named {2}.\n".
            format(model_name, save_graph_dir, save_graph_name))


if __name__ == '__main__':
    global test_args
    test_args, remaining_args = parse_args()
    generate_dot_for_model(test_args.model_path, test_args.save_graph_dir,
                           test_args.save_graph_name)
