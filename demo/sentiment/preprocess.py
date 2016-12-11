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
import sys
import random
import operator
import numpy as np
from subprocess import Popen, PIPE
from os.path import join as join_path
from optparse import OptionParser

from paddle.utils.preprocess_util import *
"""
Usage: run following command to show help message.
  python preprocess.py -h 
"""


def save_dict(dict, filename, is_reverse=True):
    """
    Save dictionary into file.
    dict:   input dictionary.
    filename: output file name, string.
    is_reverse: True, descending order by value.
                False, ascending order by value.
    """
    f = open(filename, 'w')
    for k, v in sorted(dict.items(), key=operator.itemgetter(1),\
                       reverse=is_reverse):
        f.write('%s\t%s\n' % (k, v))
    f.close()


def tokenize(sentences):
    """
    Use tokenizer.perl to tokenize input sentences.
    tokenizer.perl is tool of Moses.
    sentences : a list of input sentences.
    return: a list of processed text.
    """
    dir = './data/mosesdecoder-master/scripts/tokenizer/tokenizer.perl'
    tokenizer_cmd = [dir, '-l', 'en', '-q', '-']
    assert isinstance(sentences, list)
    text = "\n".join(sentences)
    tokenizer = Popen(tokenizer_cmd, stdin=PIPE, stdout=PIPE)
    tok_text, _ = tokenizer.communicate(text)
    toks = tok_text.split('\n')[:-1]
    return toks


def read_lines(path):
    """
    path: String, file path.
    return a list of sequence.
    """
    seqs = []
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if len(line):
                seqs.append(line)
    return seqs


class SentimentDataSetCreate():
    """
    A class to process data for sentiment analysis task.
    """

    def __init__(self,
                 data_path,
                 output_path,
                 use_okenizer=True,
                 multi_lines=False):
        """
        data_path: string, traing and testing dataset path
        output_path: string, output path, store processed dataset
        multi_lines: whether a file has multi lines.
                     In order to shuffle fully, it needs to read all files into
                     memory, then shuffle them if one file has multi lines.
        """
        self.output_path = output_path
        self.data_path = data_path

        self.train_dir = 'train'
        self.test_dir = 'test'

        self.train_list = "train.list"
        self.test_list = "test.list"

        self.label_list = "labels.list"
        self.classes_num = 0

        self.batch_size = 50000
        self.batch_dir = 'batches'

        self.dict_file = "dict.txt"
        self.dict_with_test = False
        self.dict_size = 0
        self.word_count = {}

        self.tokenizer = use_okenizer
        self.overwrite = False

        self.multi_lines = multi_lines

        self.train_dir = join_path(data_path, self.train_dir)
        self.test_dir = join_path(data_path, self.test_dir)
        self.train_list = join_path(output_path, self.train_list)
        self.test_list = join_path(output_path, self.test_list)
        self.label_list = join_path(output_path, self.label_list)
        self.dict_file = join_path(output_path, self.dict_file)

    def data_list(self, path):
        """
        create dataset from path
        path: data path
        return: data list
        """
        label_set = get_label_set_from_dir(path)
        data = []
        for lab_name in label_set.keys():
            file_paths = list_files(join_path(path, lab_name))
            for p in file_paths:
                data.append({"label"  : label_set[lab_name],\
                             "seq_path": p})
        return data, label_set

    def create_dict(self, data):
        """
        create dict for input data.
        data: list, [sequence, sequnce, ...]
        """
        for seq in data:
            for w in seq.strip().lower().split():
                if w not in self.word_count:
                    self.word_count[w] = 1
                else:
                    self.word_count[w] += 1

    def create_dataset(self):
        """
        create file batches and dictionary of train data set.
        If the self.overwrite is false and train.list already exists in
        self.output_path, this function will not create and save file
        batches from the data set path.
        return: dictionary size, class number.
        """
        out_path = self.output_path
        if out_path and not os.path.exists(out_path):
            os.makedirs(out_path)

        # If self.overwrite is false or self.train_list has existed,
        # it will not process dataset.
        if not (self.overwrite or not os.path.exists(self.train_list)):
            print "%s already exists." % self.train_list
            return

        # Preprocess train data.
        train_data, train_lab_set = self.data_list(self.train_dir)
        print "processing train set..."
        file_lists = self.save_data(train_data, "train", self.batch_size, True,
                                    True)
        save_list(file_lists, self.train_list)

        # If have test data path, preprocess test data.
        if os.path.exists(self.test_dir):
            test_data, test_lab_set = self.data_list(self.test_dir)
            assert (train_lab_set == test_lab_set)
            print "processing test set..."
            file_lists = self.save_data(test_data, "test", self.batch_size,
                                        False, self.dict_with_test)
            save_list(file_lists, self.test_list)

        # save labels set.
        save_dict(train_lab_set, self.label_list, False)
        self.classes_num = len(train_lab_set.keys())

        # save dictionary.
        save_dict(self.word_count, self.dict_file, True)
        self.dict_size = len(self.word_count)

    def save_data(self,
                  data,
                  prefix="",
                  batch_size=50000,
                  is_shuffle=False,
                  build_dict=False):
        """
        Create batches for a Dataset object.
        data: the Dataset object to process.
        prefix: the prefix of each batch.
        batch_size: number of data in each batch.
        build_dict: whether to build dictionary for data

        return: list of batch names
        """
        if is_shuffle and self.multi_lines:
            return self.save_data_multi_lines(data, prefix, batch_size,
                                              build_dict)

        if is_shuffle:
            random.shuffle(data)
        num_batches = int(math.ceil(len(data) / float(batch_size)))
        batch_names = []
        for i in range(num_batches):
            batch_name = join_path(self.output_path,
                                   "%s_part_%03d" % (prefix, i))
            begin = i * batch_size
            end = min((i + 1) * batch_size, len(data))
            # read a batch of data
            label_list, data_list = self.get_data_list(begin, end, data)
            if build_dict:
                self.create_dict(data_list)
            self.save_file(label_list, data_list, batch_name)
            batch_names.append(batch_name)

        return batch_names

    def get_data_list(self, begin, end, data):
        """
        begin: int, begining index of data.
        end: int, ending index of data.
        data: a list of {"seq_path": seqquence path, "label": label index}

        return a list of label and a list of sequence.
        """
        label_list = []
        data_list = []
        for j in range(begin, end):
            seqs = read_lines(data[j]["seq_path"])
            lab = int(data[j]["label"])
            #File may have multiple lines.
            for seq in seqs:
                data_list.append(seq)
                label_list.append(lab)
        if self.tokenizer:
            data_list = tokenize(data_list)
        return label_list, data_list

    def save_data_multi_lines(self,
                              data,
                              prefix="",
                              batch_size=50000,
                              build_dict=False):
        """
        In order to shuffle fully, there is no need to load all data if
        each file only contains one sample, it only needs to shuffle list
        of file name. But one file contains multi lines, each line is one
        sample. It needs to read all data into memory to shuffle fully.
        This interface is mainly for data containning multi lines in each
        file, which consumes more memory if there is a great mount of data.

        data: the Dataset object to process.
        prefix: the prefix of each batch.
        batch_size: number of data in each batch.
        build_dict: whether to build dictionary for data

        return: list of batch names
        """
        assert self.multi_lines
        label_list = []
        data_list = []

        # read all data
        label_list, data_list = self.get_data_list(0, len(data), data)
        if build_dict:
            self.create_dict(data_list)

        length = len(label_list)
        perm_list = np.array([i for i in xrange(length)])
        random.shuffle(perm_list)

        num_batches = int(math.ceil(length / float(batch_size)))
        batch_names = []
        for i in range(num_batches):
            batch_name = join_path(self.output_path,
                                   "%s_part_%03d" % (prefix, i))
            begin = i * batch_size
            end = min((i + 1) * batch_size, length)
            sub_label = [label_list[perm_list[i]] for i in range(begin, end)]
            sub_data = [data_list[perm_list[i]] for i in range(begin, end)]
            self.save_file(sub_label, sub_data, batch_name)
            batch_names.append(batch_name)

        return batch_names

    def save_file(self, label_list, data_list, filename):
        """
        Save data into file.
        label_list: a list of int value.
        data_list: a list of sequnece.
        filename: output file name.
        """
        f = open(filename, 'w')
        print "saving file: %s" % filename
        for lab, seq in zip(label_list, data_list):
            f.write('%s\t\t%s\n' % (lab, seq))
        f.close()


def option_parser():
    parser = OptionParser(usage="usage: python preprcoess.py "\
                                "-i data_dir [options]")
    parser.add_option(
        "-i",
        "--data",
        action="store",
        dest="input",
        help="Input data directory.")
    parser.add_option(
        "-o",
        "--output",
        action="store",
        dest="output",
        default=None,
        help="Output directory.")
    parser.add_option(
        "-t",
        "--tokenizer",
        action="store",
        dest="use_tokenizer",
        default=True,
        help="Whether to use tokenizer.")
    parser.add_option("-m", "--multi_lines", action="store",
                      dest="multi_lines", default=False,
                      help="If input text files have multi lines and they "\
                           "need to be shuffled, you should set -m True,")
    return parser.parse_args()


def main():
    options, args = option_parser()
    data_dir = options.input
    output_dir = options.output
    use_tokenizer = options.use_tokenizer
    multi_lines = options.multi_lines
    if output_dir is None:
        outname = os.path.basename(options.input)
        output_dir = join_path(os.path.dirname(data_dir), 'pre-' + outname)
    data_creator = SentimentDataSetCreate(data_dir, output_dir, use_tokenizer,
                                          multi_lines)
    data_creator.create_dataset()


if __name__ == '__main__':
    main()
