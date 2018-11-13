# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import argparse

__all__ = ['parse_args', ]

BENCHMARK_MODELS = [
    "machine_translation", "resnet", "se_resnext", "vgg", "mnist",
    "stacked_dynamic_lstm", "resnet_with_preprocess"
]


def parse_args():
    parser = argparse.ArgumentParser('Fluid model benchmarks.')
    parser.add_argument(
        '--model',
        type=str,
        choices=BENCHMARK_MODELS,
        default='resnet',
        help='The model to run benchmark with.')
    parser.add_argument(
        '--batch_size', type=int, default=32, help='The minibatch size.')
    #  args related to learning rate
    parser.add_argument(
        '--learning_rate', type=float, default=0.001, help='The learning rate.')
    # TODO(wuyi): add "--use_fake_data" option back.
    parser.add_argument(
        '--skip_batch_num',
        type=int,
        default=5,
        help='The first num of minibatch num to skip, for better performance test'
    )
    parser.add_argument(
        '--iterations', type=int, default=80, help='The number of minibatches.')
    parser.add_argument(
        '--pass_num', type=int, default=100, help='The number of passes.')
    parser.add_argument(
        '--data_format',
        type=str,
        default='NCHW',
        choices=['NCHW', 'NHWC'],
        help='The data data_format, now only support NCHW.')
    parser.add_argument(
        '--device',
        type=str,
        default='GPU',
        choices=['CPU', 'GPU'],
        help='The device type.')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='If gpus > 1, will use ParallelExecutor to run, else use Executor.')
    # this option is available only for vgg and resnet.
    parser.add_argument(
        '--cpus',
        type=int,
        default=1,
        help='If cpus > 1, will set ParallelExecutor to use multiple threads.')
    parser.add_argument(
        '--data_set',
        type=str,
        default='flowers',
        choices=['cifar10', 'flowers', 'imagenet'],
        help='Optional dataset for benchmark.')
    parser.add_argument(
        '--infer_only', action='store_true', help='If set, run forward only.')
    parser.add_argument(
        '--use_cprof', action='store_true', help='If set, use cProfile.')
    parser.add_argument(
        '--use_nvprof',
        action='store_true',
        help='If set, use nvprof for CUDA.')
    parser.add_argument(
        '--no_test',
        action='store_true',
        help='If set, do not test the testset during training.')
    parser.add_argument(
        '--memory_optimize',
        action='store_true',
        help='If set, optimize runtime memory before start.')
    parser.add_argument(
        '--use_fake_data',
        action='store_true',
        help='If set ommit the actual read data operators.')
    parser.add_argument(
        '--profile', action='store_true', help='If set, profile a few steps.')
    parser.add_argument(
        '--update_method',
        type=str,
        default='local',
        choices=['local', 'pserver', 'nccl2'],
        help='Choose parameter update method, can be local, pserver, nccl2.')
    parser.add_argument(
        '--no_split_var',
        action='store_true',
        default=False,
        help='Whether split variables into blocks when update_method is pserver')
    parser.add_argument(
        '--async_mode',
        action='store_true',
        default=False,
        help='Whether start pserver in async mode to support ASGD')
    parser.add_argument(
        '--use_reader_op',
        action='store_true',
        help='Whether to use reader op, and must specify the data path if set this to true.'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default="",
        help='Directory that contains all the training recordio files.')
    parser.add_argument(
        '--test_data_path',
        type=str,
        default="",
        help='Directory that contains all the test data (NOT recordio).')
    parser.add_argument(
        '--use_inference_transpiler',
        action='store_true',
        help='If set, use inference transpiler to optimize the program.')
    parser.add_argument(
        '--no_random',
        action='store_true',
        help='If set, keep the random seed and do not shuffle the data.')
    parser.add_argument(
        '--reduce_strategy',
        type=str,
        choices=['reduce', 'all_reduce'],
        default='all_reduce',
        help='Specify the reduce strategy, can be reduce, all_reduce')
    parser.add_argument(
        '--fuse_broadcast_op',
        action='store_true',
        help='If set, would fuse multiple broadcast operators into one fused_broadcast operator.'
    )
    args = parser.parse_args()
    return args
