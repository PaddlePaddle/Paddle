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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics import Metric
from model import Model, Input, Loss, set_device
from text import SequenceTagging
from reader import LacDataset, create_lexnet_data_generator, create_dataloader 

import paddle.fluid as fluid
from paddle.fluid.optimizer import AdamOptimizer


class SeqTagging(Model):
    def __init__(self, args, vocab_size, num_labels, length=None):
        super(SeqTagging, self).__init__()
        """
        define the lexical analysis network structure
        word: stores the input of the model
        for_infer: a boolean value, indicating if the model to be created is for training or predicting.

        return:
            for infer: return the prediction
            otherwise: return the prediction
        """
        self.word_emb_dim = args.word_emb_dim
        self.vocab_size = vocab_size
        self.num_labels = num_labels
        self.grnn_hidden_dim = args.grnn_hidden_dim
        self.emb_lr = args.emb_learning_rate if 'emb_learning_rate' in dir(
            args) else 1.0
        self.crf_lr = args.emb_learning_rate if 'crf_learning_rate' in dir(
            args) else 1.0
        self.bigru_num = args.bigru_num
        self.batch_size = args.batch_size
        self.init_bound = 0.1
        self.length=length

        self.sequence_tagging = SequenceTagging(
                        vocab_size=self.vocab_size,
                        num_labels=self.num_labels,
                        batch_size=self.batch_size,
                        word_emb_dim=self.word_emb_dim,
                        grnn_hidden_dim=self.grnn_hidden_dim,
                        emb_learning_rate=self.emb_lr,
                        crf_learning_rate=self.crf_lr,
                        bigru_num=self.bigru_num,
                        init_bound=self.init_bound,
                        length=self.length)

    def forward(self, word, target, lengths):
        """
        Configure the network
        """
        crf_decode, avg_cost, lengths = self.sequence_tagging(word, target, lengths)
        return crf_decode, avg_cost, lengths


class Chunk_eval(fluid.dygraph.Layer):
    def __init__(self,
                 num_chunk_types,
                 chunk_scheme,
                 excluded_chunk_types=None):
        super(Chunk_eval, self).__init__()
        self.num_chunk_types = num_chunk_types
        self.chunk_scheme = chunk_scheme
        self.excluded_chunk_types = excluded_chunk_types

    def forward(self, input, label, seq_length=None):
        precision = self._helper.create_variable_for_type_inference(
            dtype="float32")
        recall = self._helper.create_variable_for_type_inference(
            dtype="float32")
        f1_score = self._helper.create_variable_for_type_inference(
            dtype="float32")
        num_infer_chunks = self._helper.create_variable_for_type_inference(
            dtype="int64")
        num_label_chunks = self._helper.create_variable_for_type_inference(
            dtype="int64")
        num_correct_chunks = self._helper.create_variable_for_type_inference(
            dtype="int64")

        this_input = {"Inference": input, "Label": label[0]}
        if seq_length:
            this_input["SeqLength"] = seq_length[0]
        self._helper.append_op(
            type='chunk_eval',
            inputs=this_input,
            outputs={
                "Precision": [precision],
                "Recall": [recall],
                "F1-Score": [f1_score],
                "NumInferChunks": [num_infer_chunks],
                "NumLabelChunks": [num_label_chunks],
                "NumCorrectChunks": [num_correct_chunks]
            },
            attrs={
                "num_chunk_types": self.num_chunk_types,
                "chunk_scheme": self.chunk_scheme,
                "excluded_chunk_types": self.excluded_chunk_types or []
            })
        return (num_infer_chunks, num_label_chunks, num_correct_chunks)


class LacLoss(Loss):
    def __init__(self):
        super(LacLoss, self).__init__()
        pass

    def forward(self, outputs, labels):
        avg_cost = outputs[1]
        return avg_cost


class ChunkEval(Metric):
    def __init__(self, num_labels, name=None, *args, **kwargs):
        super(ChunkEval, self).__init__(*args, **kwargs)
        self._init_name(name)
        self.chunk_eval = Chunk_eval(
            int(math.ceil((num_labels - 1) / 2.0)), "IOB")
        self.reset()

    def add_metric_op(self, pred, label, *args, **kwargs):
        crf_decode = pred[0]
        lengths = pred[2]
        (num_infer_chunks, num_label_chunks,
         num_correct_chunks) = self.chunk_eval(
             input=crf_decode, label=label, seq_length=lengths)
        return [num_infer_chunks, num_label_chunks, num_correct_chunks]

    def update(self, num_infer_chunks, num_label_chunks, num_correct_chunks,
               *args, **kwargs):
        self.infer_chunks_total += num_infer_chunks
        self.label_chunks_total += num_label_chunks
        self.correct_chunks_total += num_correct_chunks
        precision = float(
            num_correct_chunks) / num_infer_chunks if num_infer_chunks else 0
        recall = float(
            num_correct_chunks) / num_label_chunks if num_label_chunks else 0
        f1_score = float(2 * precision * recall) / (
            precision + recall) if num_correct_chunks else 0
        return [precision, recall, f1_score]

    def reset(self):
        self.infer_chunks_total = 0
        self.label_chunks_total = 0
        self.correct_chunks_total = 0

    def accumulate(self):
        precision = float(
            self.correct_chunks_total
        ) / self.infer_chunks_total if self.infer_chunks_total else 0
        recall = float(
            self.correct_chunks_total
        ) / self.label_chunks_total if self.label_chunks_total else 0
        f1_score = float(2 * precision * recall) / (
            precision + recall) if self.correct_chunks_total else 0
        res = [precision, recall, f1_score]
        return res

    def _init_name(self, name):
        name = name or 'chunk eval'
        self._name = ['precision', 'recall', 'F1']

    def name(self):
        return self._name


def main(args):
    place = set_device(args.device)
    fluid.enable_dygraph(place) if args.dynamic else None

    inputs = [
        Input(
            [None, args.max_seq_len], 'int64', name='words'), Input(
                [None, args.max_seq_len], 'int64', name='target'), Input(
                    [None], 'int64', name='length')
    ]
    labels = [Input([None, args.max_seq_len], 'int64', name='labels')]

    feed_list = None if args.dynamic else [x.forward() for x in inputs + labels]
    dataset = LacDataset(args)
    train_path = os.path.join(args.data, "train.tsv")
    test_path = os.path.join(args.data, "test.tsv")

    train_generator = create_lexnet_data_generator(
        args, reader=dataset, file_name=train_path, place=place, mode="train")
    test_generator = create_lexnet_data_generator(
        args, reader=dataset, file_name=test_path, place=place, mode="test")

    train_dataset = create_dataloader(
        train_generator, place, feed_list=feed_list)
    test_dataset = create_dataloader(
        test_generator, place, feed_list=feed_list)

    vocab_size = dataset.vocab_size
    num_labels = dataset.num_labels
    model = SeqTagging(args, vocab_size, num_labels)

    optim = AdamOptimizer(
        learning_rate=args.base_learning_rate,
        parameter_list=model.parameters())

    model.prepare(
        optim,
        LacLoss(),
        ChunkEval(num_labels),
        inputs=inputs,
        labels=labels,
        device=args.device)

    if args.resume is not None:
        model.load(args.resume)

    model.fit(train_dataset,
              test_dataset,
              epochs=args.epoch,
              batch_size=args.batch_size,
              eval_freq=args.eval_freq,
              save_freq=args.save_freq,
              save_dir=args.save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("LAC training")
    parser.add_argument(
        "-dir", "--data", default=None, type=str, help='path to LAC dataset')
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
        "-r",
        "--resume",
        default=None,
        type=str,
        help="checkpoint path to resume")
    parser.add_argument(
        "-o",
        "--save_dir",
        default="./model",
        type=str,
        help="save model path")
    parser.add_argument(
        "-sf", "--save_freq", default=1, type=int, help="save frequency")
    parser.add_argument(
        "-ef", "--eval_freq", default=1, type=int, help="eval frequency")

    args = parser.parse_args()
    print(args)
    main(args)
