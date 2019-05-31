# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.framework import IrGraph, Variable, Program
import paddle
from reader import *
import argparse
from quantization_mkldnn_pass import *


def parse_args():
    parser = argparse.ArgumentParser('Convolution model benchmark.')
    parser.add_argument(
        '--batch_size', type=int, default=1, help='The minibatch size.')

    parser.add_argument(
        '--iterations',
        type=int,
        default=100,
        help='The number of minibatches to process. 0')

    parser.add_argument(
        '--infer_model_path',
        type=str,
        default='',
        help='The directory for loading inference model.')

    parser.add_argument(
        '--data_dir',
        type=str,
        default='small_data/ILSVRC2012',
        help='A directory with test data files.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    model_path = args.infer_model_path
    [inference_program, feed_target_names,
     fetch_targets] = fluid.io.load_inference_model(model_path, exe)
    graph = IrGraph(core.Graph(inference_program.desc), for_test=True)
    scope = fluid.executor.global_scope()
    transform_pass = TransformForMkldnnPass(scope, place)
    graph_transformed = transform_pass.apply(graph)
    program = graph_transformed.to_program()

    batch_size = args.batch_size
    val_reader = paddle.batch(val(args.data_dir), batch_size)
    iterations = 1
    dshape = [3, 224, 224]
    image_shape = [3, 224, 224]
    total = 0
    correct = 0
    correct_5 = 0
    for batch_id, data in enumerate(val_reader()):
        image = np.array(map(lambda x: x[0].reshape(dshape), data)).astype(
            'float32')
        label = np.array([x[1] for x in data]).astype("int64")
        label = label.reshape([-1, 1])
        out = exe.run(program,
                      feed={feed_target_names[0]: image},
                      fetch_list=fetch_targets)
        result = np.array(out[0][0])
        index = result.argsort()
        top_1_index = index[-1]
        top_5_index = index[-5:]
        total += 1
        if top_1_index == label:
            correct += 1
        if label in top_5_index:
            correct_5 += 1
        acc1 = float(correct) / float(total)
        acc5 = float(correct_5) / float(total)
        if batch_id % 10 == 0:
            print("Testbatch {0}, "
                "acc1 {1}, acc5 {2}".format(batch_id, \
                 acc1, acc5))
            sys.stdout.flush()

        if batch_id == args.iterations:
            break

    total_acc1 = float(correct) / float(total)
    total_acc5 = float(correct_5) / float(total)
    print("End test: test_acc1 {0}, test_acc5 {1}".format(total_acc1,
                                                          total_acc5))
    sys.stdout.flush()
