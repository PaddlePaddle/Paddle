#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""
SequenceTagging network structure
"""

from __future__ import division
from __future__ import print_function

import io
import os
import sys
import math
import argparse
import numpy as np

from train import SeqTagging
from utils.check import check_gpu, check_version
from reader import LacDataset, create_lexnet_data_generator, create_dataloader

work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(work_dir)
from hapi.model import set_device, Input

import paddle.fluid as fluid
from paddle.fluid.optimizer import AdamOptimizer
from paddle.fluid.layers.utils import flatten


def main(args):
    place = set_device(args.device)
    fluid.enable_dygraph(place) if args.dynamic else None

    inputs = [Input([None, args.max_seq_len], 'int64', name='words'), 
              Input([None], 'int64', name='length')]

    feed_list = None if args.dynamic else [x.forward() for x in inputs]
    dataset = LacDataset(args)
    predict_path = args.predict_file

    predict_generator = create_lexnet_data_generator(
        args, reader=dataset, file_name=predict_path, place=place, mode="predict")

    predict_dataset = create_dataloader(
        predict_generator, place, feed_list=feed_list)

    vocab_size = dataset.vocab_size
    num_labels = dataset.num_labels
    model = SeqTagging(args, vocab_size, num_labels)

    optim = AdamOptimizer(
        learning_rate=args.base_learning_rate,
        parameter_list=model.parameters())

    model.mode = "test"
    model.prepare(inputs=inputs)

    model.load(args.init_from_checkpoint)

    f = open(args.output_file, "wb")
    for data in predict_dataset(): 
        results, length = model.test(inputs=flatten(data))
        #length_list = np.fromstring(length, dtype=str)
        for i in range(len(results)): 
            word_len = length[i]
            word_ids = results[i][: word_len]
            tags = [dataset.id2label_dict[str(id)] for id in word_ids]
            f.write("\002".join(tags) + "\n")
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser("sequence tagging training")
    parser.add_argument(
        "-wd",
        "--word_dict_path",
        default=None,
        type=str,
        help='word dict path')
    parser.add_argument(
        "-ld",
        "--label_dict_path",
        default=None,
        type=str,
        help='label dict path')
    parser.add_argument(
        "-wrd",
        "--word_rep_dict_path",
        default=None,
        type=str,
        help='The path of the word replacement Dictionary.')
    parser.add_argument(
        "-dev",
        "--device",
        type=str,
        default='gpu',
        help="device to use, gpu or cpu")
    parser.add_argument(
        "-d", "--dynamic", action='store_true', help="enable dygraph mode")
    parser.add_argument(
        "-e", "--epoch", default=10, type=int, help="number of epoch")
    parser.add_argument(
        '-lr',
        '--base_learning_rate',
        default=1e-3,
        type=float,
        metavar='LR',
        help='initial learning rate')
    parser.add_argument(
        "--word_emb_dim",
        default=128,
        type=int,
        help='word embedding dimension')
    parser.add_argument(
        "--grnn_hidden_dim", default=128, type=int, help="hidden dimension")
    parser.add_argument(
        "--bigru_num", default=2, type=int, help='the number of bi-rnn')
    parser.add_argument("-elr", "--emb_learning_rate", default=1.0, type=float)
    parser.add_argument("-clr", "--crf_learning_rate", default=1.0, type=float)
    parser.add_argument(
        "-b", "--batch_size", default=300, type=int, help="batch size")
    parser.add_argument(
        "--max_seq_len", default=126, type=int, help="max sequence length")
    parser.add_argument(
        "-n", "--num_devices", default=1, type=int, help="number of devices")
    parser.add_argument(
        "-o",
        "--save_dir",
        default="./model",
        type=str,
        help="save model path")
    parser.add_argument(
        "--init_from_checkpoint",
        default=None,
        type=str,
        help="load init model parameters")
    parser.add_argument(
        "--init_from_pretrain_model",
        default=None,
        type=str,
        help="load pretrain model parameters")
    parser.add_argument(
        "-sf", "--save_freq", default=1, type=int, help="save frequency")
    parser.add_argument(
        "-ef", "--eval_freq", default=1, type=int, help="eval frequency")
    parser.add_argument(
        "--output_file", default="predict.result", type=str, help="predict output file")
    parser.add_argument(
        "--predict_file", default="./data/infer.tsv", type=str, help="predict output file")
    parser.add_argument(
        "--mode", default="train", type=str, help="train|test|predict")

    args = parser.parse_args()
    print(args)
    check_gpu(args.device)
    check_version()

    main(args)
