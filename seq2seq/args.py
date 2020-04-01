# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import distutils.util


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train_data_prefix", type=str, help="file prefix for train data")
    parser.add_argument(
        "--eval_data_prefix", type=str, help="file prefix for eval data")
    parser.add_argument(
        "--test_data_prefix", type=str, help="file prefix for test data")
    parser.add_argument(
        "--vocab_prefix", type=str, help="file prefix for vocab")
    parser.add_argument("--src_lang", type=str, help="source language suffix")
    parser.add_argument("--tar_lang", type=str, help="target language suffix")

    parser.add_argument(
        "--attention",
        type=eval,
        default=False,
        help="Whether use attention model")

    parser.add_argument(
        "--optimizer",
        type=str,
        default='adam',
        help="optimizer to use, only supprt[sgd|adam]")

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="learning rate for optimizer")

    parser.add_argument(
        "--num_layers",
        type=int,
        default=1,
        help="layers number of encoder and decoder")
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=100,
        help="hidden size of encoder and decoder")
    parser.add_argument("--src_vocab_size", type=int, help="source vocab size")
    parser.add_argument("--tar_vocab_size", type=int, help="target vocab size")

    parser.add_argument(
        "--batch_size", type=int, help="batch size of each step")

    parser.add_argument(
        "--max_epoch", type=int, default=12, help="max epoch for the training")

    parser.add_argument(
        "--max_len",
        type=int,
        default=50,
        help="max length for source and target sentence")
    parser.add_argument(
        "--dropout", type=float, default=0.0, help="drop probability")
    parser.add_argument(
        "--init_scale",
        type=float,
        default=0.0,
        help="init scale for parameter")
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=5.0,
        help="max grad norm for global norm clip")

    parser.add_argument(
        "--model_path",
        type=str,
        default='model',
        help="model path for model to save")

    parser.add_argument(
        "--reload_model", type=str, help="reload model to inference")

    parser.add_argument(
        "--infer_file", type=str, help="file name for inference")
    parser.add_argument(
        "--infer_output_file",
        type=str,
        default='infer_output',
        help="file name for inference output")
    parser.add_argument(
        "--beam_size", type=int, default=10, help="file name for inference")

    parser.add_argument(
        '--use_gpu',
        type=eval,
        default=False,
        help='Whether using gpu [True|False]')

    parser.add_argument(
        '--eager_run', type=eval, default=False, help='Whether to use dygraph')

    parser.add_argument(
        "--enable_ce",
        action='store_true',
        help="The flag indicating whether to run the task "
        "for continuous evaluation.")

    parser.add_argument(
        "--profile", action='store_true', help="Whether enable the profile.")
    # NOTE: profiler args, used for benchmark
    parser.add_argument(
        "--profiler_path",
        type=str,
        default='./seq2seq.profile',
        help="the profiler output file path. (used for benchmark)")
    args = parser.parse_args()
    return args
