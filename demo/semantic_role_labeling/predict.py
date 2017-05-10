# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

import os
import numpy as np
from optparse import OptionParser
from py_paddle import swig_paddle, DataProviderConverter
from paddle.trainer.PyDataProvider2 import integer_value_sequence
from paddle.trainer.config_parser import parse_config
"""
Usage: run following command to show help message.
  python predict.py -h
"""
UNK_IDX = 0


class Prediction():
    def __init__(self, train_conf, dict_file, model_dir, label_file,
                 predicate_dict_file):
        """
        train_conf: trainer configure.
        dict_file: word dictionary file name.
        model_dir: directory of model.
        """

        self.dict = {}
        self.labels = {}
        self.predicate_dict = {}
        self.labels_reverse = {}
        self.load_dict_label(dict_file, label_file, predicate_dict_file)

        len_dict = len(self.dict)
        len_label = len(self.labels)
        len_pred = len(self.predicate_dict)

        conf = parse_config(
            train_conf, 'dict_len=' + str(len_dict) + ',label_len=' +
            str(len_label) + ',pred_len=' + str(len_pred) + ',is_predict=True')
        self.network = swig_paddle.GradientMachine.createFromConfigProto(
            conf.model_config)
        self.network.loadParameters(model_dir)

        slots = [
            integer_value_sequence(len_dict), integer_value_sequence(len_dict),
            integer_value_sequence(len_dict), integer_value_sequence(len_dict),
            integer_value_sequence(len_dict), integer_value_sequence(len_dict),
            integer_value_sequence(len_pred), integer_value_sequence(2)
        ]
        self.converter = DataProviderConverter(slots)

    def load_dict_label(self, dict_file, label_file, predicate_dict_file):
        """
        Load dictionary from self.dict_file.
        """
        for line_count, line in enumerate(open(dict_file, 'r')):
            self.dict[line.strip()] = line_count

        for line_count, line in enumerate(open(label_file, 'r')):
            self.labels[line.strip()] = line_count
            self.labels_reverse[line_count] = line.strip()

        for line_count, line in enumerate(open(predicate_dict_file, 'r')):
            self.predicate_dict[line.strip()] = line_count

    def get_data(self, data_file):
        """
        Get input data of paddle format.
        """
        with open(data_file, 'r') as fdata:
            for line in fdata:
                sentence, predicate, ctx_n2, ctx_n1, ctx_0, ctx_p1, ctx_p2, mark, label = line.strip(
                ).split('\t')
                words = sentence.split()
                sen_len = len(words)

                word_slot = [self.dict.get(w, UNK_IDX) for w in words]
                predicate_slot = [self.predicate_dict.get(predicate, UNK_IDX)
                                  ] * sen_len
                ctx_n2_slot = [self.dict.get(ctx_n2, UNK_IDX)] * sen_len
                ctx_n1_slot = [self.dict.get(ctx_n1, UNK_IDX)] * sen_len
                ctx_0_slot = [self.dict.get(ctx_0, UNK_IDX)] * sen_len
                ctx_p1_slot = [self.dict.get(ctx_p1, UNK_IDX)] * sen_len
                ctx_p2_slot = [self.dict.get(ctx_p2, UNK_IDX)] * sen_len

                marks = mark.split()
                mark_slot = [int(w) for w in marks]

                yield word_slot, ctx_n2_slot, ctx_n1_slot, \
                      ctx_0_slot, ctx_p1_slot, ctx_p2_slot, predicate_slot, mark_slot

    def predict(self, data_file, output_file):
        """
        data_file: file name of input data.
        """
        input = self.converter(self.get_data(data_file))
        output = self.network.forwardTest(input)
        lab = output[0]["id"].tolist()

        with open(data_file, 'r') as fin, open(output_file, 'w') as fout:
            index = 0
            for line in fin:
                sen = line.split('\t')[0]
                len_sen = len(sen.split())
                line_labels = lab[index:index + len_sen]
                index += len_sen
                fout.write(sen + '\t' + ' '.join(
                    [self.labels_reverse[i] for i in line_labels]) + '\n')


def option_parser():
    usage = (
        "python predict.py -c config -w model_dir "
        "-d word dictionary -l label_file -i input_file  -p pred_dict_file")
    parser = OptionParser(usage="usage: %s [options]" % usage)
    parser.add_option(
        "-c",
        "--tconf",
        action="store",
        dest="train_conf",
        help="network config")
    parser.add_option(
        "-d",
        "--dict",
        action="store",
        dest="dict_file",
        help="dictionary file")
    parser.add_option(
        "-l",
        "--label",
        action="store",
        dest="label_file",
        default=None,
        help="label file")
    parser.add_option(
        "-p",
        "--predict_dict_file",
        action="store",
        dest="predict_dict_file",
        default=None,
        help="predict_dict_file")
    parser.add_option(
        "-i",
        "--data",
        action="store",
        dest="data_file",
        help="data file to predict")
    parser.add_option(
        "-w",
        "--model",
        action="store",
        dest="model_path",
        default=None,
        help="model path")

    parser.add_option(
        "-o",
        "--output_file",
        action="store",
        dest="output_file",
        default=None,
        help="output file")
    return parser.parse_args()


def main():
    options, args = option_parser()
    train_conf = options.train_conf
    data_file = options.data_file
    dict_file = options.dict_file
    model_path = options.model_path
    label_file = options.label_file
    predict_dict_file = options.predict_dict_file
    output_file = options.output_file

    swig_paddle.initPaddle("--use_gpu=0")
    predict = Prediction(train_conf, dict_file, model_path, label_file,
                         predict_dict_file)
    predict.predict(data_file, output_file)


if __name__ == '__main__':
    main()
