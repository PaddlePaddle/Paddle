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
lexical analysis network structure
"""

from __future__ import division
from __future__ import print_function

import io
import os
import sys
import math
import argparse
import numpy as np

from metrics import Metric
from model import Model, Input, Loss, set_device

import paddle.fluid as fluid
from paddle.fluid.optimizer import AdamOptimizer
from paddle.fluid.initializer import NormalInitializer
from paddle.fluid.dygraph.nn import Embedding, Linear, GRUUnit


class DynamicGRU(fluid.dygraph.Layer):
    def __init__(self,
                 size,
                 h_0=None,
                 param_attr=None,
                 bias_attr=None,
                 is_reverse=False,
                 gate_activation='sigmoid',
                 candidate_activation='tanh',
                 origin_mode=False,
                 init_size=None):
        super(DynamicGRU, self).__init__()

        self.gru_unit = GRUUnit(
            size * 3,
            param_attr=param_attr,
            bias_attr=bias_attr,
            activation=candidate_activation,
            gate_activation=gate_activation,
            origin_mode=origin_mode)

        self.size = size
        self.h_0 = h_0
        self.is_reverse = is_reverse

    def forward(self, inputs):
        hidden = self.h_0
        res = []

        for i in range(inputs.shape[1]):
            if self.is_reverse:
                i = inputs.shape[1] - 1 - i
            input_ = inputs[:, i:i + 1, :]
            input_ = fluid.layers.reshape(
                input_, [-1, input_.shape[2]], inplace=False)
            hidden, reset, gate = self.gru_unit(input_, hidden)
            hidden_ = fluid.layers.reshape(
                hidden, [-1, 1, hidden.shape[1]], inplace=False)
            res.append(hidden_)
        if self.is_reverse:
            res = res[::-1]
        res = fluid.layers.concat(res, axis=1)
        return res


class BiGRU(fluid.dygraph.Layer):
    def __init__(self, input_dim, grnn_hidden_dim, init_bound, h_0=None):
        super(BiGRU, self).__init__()

        self.pre_gru = Linear(
            input_dim=input_dim,
            output_dim=grnn_hidden_dim * 3,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))

        self.gru = DynamicGRU(
            size=grnn_hidden_dim,
            h_0=h_0,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))

        self.pre_gru_r = Linear(
            input_dim=input_dim,
            output_dim=grnn_hidden_dim * 3,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))

        self.gru_r = DynamicGRU(
            size=grnn_hidden_dim,
            is_reverse=True,
            h_0=h_0,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))

    def forward(self, input_feature):
        res_pre_gru = self.pre_gru(input_feature)
        res_gru = self.gru(res_pre_gru)
        res_pre_gru_r = self.pre_gru_r(input_feature)
        res_gru_r = self.gru_r(res_pre_gru_r)
        bi_merge = fluid.layers.concat(input=[res_gru, res_gru_r], axis=-1)
        return bi_merge


class Linear_chain_crf(fluid.dygraph.Layer):
    def __init__(self, param_attr, size=None, is_test=False, dtype='float32'):
        super(Linear_chain_crf, self).__init__()

        self._param_attr = param_attr
        self._dtype = dtype
        self._size = size
        self._is_test = is_test
        self._transition = self.create_parameter(
            attr=self._param_attr,
            shape=[self._size + 2, self._size],
            dtype=self._dtype)

    @property
    def weight(self):
        return self._transition

    @weight.setter
    def weight(self, value):
        self._transition = value

    def forward(self, input, label, length=None):

        alpha = self._helper.create_variable_for_type_inference(
            dtype=self._dtype)
        emission_exps = self._helper.create_variable_for_type_inference(
            dtype=self._dtype)
        transition_exps = self._helper.create_variable_for_type_inference(
            dtype=self._dtype)
        log_likelihood = self._helper.create_variable_for_type_inference(
            dtype=self._dtype)
        this_inputs = {
            "Emission": [input],
            "Transition": self._transition,
            "Label": [label]
        }
        if length:
            this_inputs['Length'] = [length]
        self._helper.append_op(
            type='linear_chain_crf',
            inputs=this_inputs,
            outputs={
                "Alpha": [alpha],
                "EmissionExps": [emission_exps],
                "TransitionExps": transition_exps,
                "LogLikelihood": log_likelihood
            },
            attrs={"is_test": self._is_test, })
        return log_likelihood


class Crf_decoding(fluid.dygraph.Layer):
    def __init__(self, param_attr, size=None, is_test=False, dtype='float32'):
        super(Crf_decoding, self).__init__()

        self._dtype = dtype
        self._size = size
        self._is_test = is_test
        self._param_attr = param_attr
        self._transition = self.create_parameter(
            attr=self._param_attr,
            shape=[self._size + 2, self._size],
            dtype=self._dtype)

    @property
    def weight(self):
        return self._transition

    @weight.setter
    def weight(self, value):
        self._transition = value

    def forward(self, input, label=None, length=None):

        viterbi_path = self._helper.create_variable_for_type_inference(
            dtype=self._dtype)
        this_inputs = {
            "Emission": [input],
            "Transition": self._transition,
            "Label": label
        }
        if length:
            this_inputs['Length'] = [length]
        self._helper.append_op(
            type='crf_decoding',
            inputs=this_inputs,
            outputs={"ViterbiPath": [viterbi_path]},
            attrs={"is_test": self._is_test, })
        return viterbi_path


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


class LAC(Model):
    def __init__(self, args, vocab_size, num_labels, length=None):
        super(LAC, self).__init__()
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
        self.init_bound = 0.1

        self.word_embedding = Embedding(
            size=[self.vocab_size, self.word_emb_dim],
            dtype='float32',
            param_attr=fluid.ParamAttr(
                learning_rate=self.emb_lr,
                name="word_emb",
                initializer=fluid.initializer.Uniform(
                    low=-self.init_bound, high=self.init_bound)))

        h_0 = fluid.layers.create_global_var(
            shape=[args.batch_size, self.grnn_hidden_dim],
            value=0.0,
            dtype='float32',
            persistable=True,
            force_cpu=True,
            name='h_0')

        self.bigru_units = []
        for i in range(self.bigru_num):
            if i == 0:
                self.bigru_units.append(
                    self.add_sublayer(
                        "bigru_units%d" % i,
                        BiGRU(
                            self.grnn_hidden_dim,
                            self.grnn_hidden_dim,
                            self.init_bound,
                            h_0=h_0)))
            else:
                self.bigru_units.append(
                    self.add_sublayer(
                        "bigru_units%d" % i,
                        BiGRU(
                            self.grnn_hidden_dim * 2,
                            self.grnn_hidden_dim,
                            self.init_bound,
                            h_0=h_0)))

        self.fc = Linear(
            input_dim=self.grnn_hidden_dim * 2,
            output_dim=self.num_labels,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-self.init_bound, high=self.init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))

        self.linear_chain_crf = Linear_chain_crf(
            param_attr=fluid.ParamAttr(
                name='linear_chain_crfw', learning_rate=self.crf_lr),
            size=self.num_labels)

        self.crf_decoding = Crf_decoding(
            param_attr=fluid.ParamAttr(
                name='crfw', learning_rate=self.crf_lr),
            size=self.num_labels)

    def forward(self, word, target, lengths):
        """
        Configure the network
        """
        word_embed = self.word_embedding(word)
        input_feature = word_embed

        for i in range(self.bigru_num):
            bigru_output = self.bigru_units[i](input_feature)
            input_feature = bigru_output

        emission = self.fc(bigru_output)

        crf_cost = self.linear_chain_crf(
            input=emission, label=target, length=lengths)
        avg_cost = fluid.layers.mean(x=crf_cost)
        self.crf_decoding.weight = self.linear_chain_crf.weight
        crf_decode = self.crf_decoding(input=emission, length=lengths)
        return crf_decode, avg_cost, lengths


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


class LacDataset(object):
    """
    Load lexical analysis dataset
    """

    def __init__(self, args):
        self.word_dict_path = args.word_dict_path
        self.label_dict_path = args.label_dict_path
        self.word_rep_dict_path = args.word_rep_dict_path
        self._load_dict()

    def _load_dict(self):
        self.word2id_dict = self.load_kv_dict(
            self.word_dict_path, reverse=True, value_func=np.int64)
        self.id2word_dict = self.load_kv_dict(self.word_dict_path)
        self.label2id_dict = self.load_kv_dict(
            self.label_dict_path, reverse=True, value_func=np.int64)
        self.id2label_dict = self.load_kv_dict(self.label_dict_path)
        if self.word_rep_dict_path is None:
            self.word_replace_dict = dict()
        else:
            self.word_replace_dict = self.load_kv_dict(self.word_rep_dict_path)

    def load_kv_dict(self,
                     dict_path,
                     reverse=False,
                     delimiter="\t",
                     key_func=None,
                     value_func=None):
        """
        Load key-value dict from file
        """
        result_dict = {}
        for line in io.open(dict_path, "r", encoding='utf8'):
            terms = line.strip("\n").split(delimiter)
            if len(terms) != 2:
                continue
            if reverse:
                value, key = terms
            else:
                key, value = terms
            if key in result_dict:
                raise KeyError("key duplicated with [%s]" % (key))
            if key_func:
                key = key_func(key)
            if value_func:
                value = value_func(value)
            result_dict[key] = value
        return result_dict

    @property
    def vocab_size(self):
        return len(self.word2id_dict.values())

    @property
    def num_labels(self):
        return len(self.label2id_dict.values())

    def get_num_examples(self, filename):
        """num of line of file"""
        return sum(1 for line in io.open(filename, "r", encoding='utf8'))

    def word_to_ids(self, words):
        """convert word to word index"""
        word_ids = []
        for word in words:
            word = self.word_replace_dict.get(word, word)
            if word not in self.word2id_dict:
                word = "OOV"
            word_id = self.word2id_dict[word]
            word_ids.append(word_id)

        return word_ids

    def label_to_ids(self, labels):
        """convert label to label index"""
        label_ids = []
        for label in labels:
            if label not in self.label2id_dict:
                label = "O"
            label_id = self.label2id_dict[label]
            label_ids.append(label_id)
        return label_ids

    def file_reader(self,
                    filename,
                    mode="train",
                    batch_size=32,
                    max_seq_len=126):
        """
        yield (word_idx, target_idx) one by one from file,
            or yield (word_idx, ) in `infer` mode
        """

        def wrapper():
            fread = io.open(filename, "r", encoding="utf-8")
            headline = next(fread)
            headline = headline.strip().split('\t')
            assert len(headline) == 2 and headline[0] == "text_a" and headline[
                1] == "label"
            buf = []
            for line in fread:
                words, labels = line.strip("\n").split("\t")
                if len(words) < 1:
                    continue
                word_ids = self.word_to_ids(words.split("\002"))
                label_ids = self.label_to_ids(labels.split("\002"))
                assert len(word_ids) == len(label_ids)
                word_ids = word_ids[0:max_seq_len]
                words_len = np.int64(len(word_ids))
                word_ids += [0 for _ in range(max_seq_len - words_len)]
                label_ids = label_ids[0:max_seq_len]
                label_ids += [0 for _ in range(max_seq_len - words_len)]
                assert len(word_ids) == len(label_ids)
                yield word_ids, label_ids, words_len
            fread.close()

        return wrapper


def create_lexnet_data_generator(args, reader, file_name, place, mode="train"):
    def wrapper():
        batch_words, batch_labels, seq_lens = [], [], []
        for epoch in xrange(args.epoch):
            for instance in reader.file_reader(
                    file_name, mode, max_seq_len=args.max_seq_len)():
                words, labels, words_len = instance
                if len(seq_lens) < args.batch_size:
                    batch_words.append(words)
                    batch_labels.append(labels)
                    seq_lens.append(words_len)
                if len(seq_lens) == args.batch_size:
                    yield batch_words, batch_labels, seq_lens, batch_labels
                    batch_words, batch_labels, seq_lens = [], [], []

        if len(seq_lens) > 0:
            yield batch_words, batch_labels, seq_lens, batch_labels
            batch_words, batch_labels, seq_lens = [], [], []

    return wrapper


def create_dataloader(generator, place, feed_list=None):
    if not feed_list:
        data_loader = fluid.io.DataLoader.from_generator(
            capacity=50,
            use_double_buffer=True,
            iterable=True,
            return_list=True)
    else:
        data_loader = fluid.io.DataLoader.from_generator(
            feed_list=feed_list,
            capacity=50,
            use_double_buffer=True,
            iterable=True,
            return_list=True)
    data_loader.set_batch_generator(generator, places=place)
    return data_loader


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

    feed = [x.forward() for x in inputs + labels]
    dataset = LacDataset(args)
    train_path = os.path.join(args.data, "train.tsv")
    test_path = os.path.join(args.data, "test.tsv")

    if args.dynamic:
        feed_list = None
    else:
        feed_list = feed
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
    model = LAC(args, vocab_size, num_labels)

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
