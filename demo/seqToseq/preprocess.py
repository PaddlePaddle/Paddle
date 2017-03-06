#!/bin/env python
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
"""
Example:
    python preprocess.py -i INPUT [-d DICTSIZE] [-m]

Options:
    -h, --help     show this help message and exit
    -i INPUT       input original dataset path
    -d DICTSIZE    specified word count of dictionary
    -m --mergeDict merge source and target dictionary
"""
import os
import sys

import string
from optparse import OptionParser
from paddle.utils.preprocess_util import save_list, DatasetCreater


class SeqToSeqDatasetCreater(DatasetCreater):
    """
    A class to process data for sequence to sequence application.
    """

    def __init__(self, data_path, output_path):
        """
        data_path: the path to store the train data, test data and gen data
        output_path: the path to store the processed dataset
        """
        DatasetCreater.__init__(self, data_path)
        self.gen_dir_name = 'gen'
        self.gen_list_name = 'gen.list'
        self.output_path = output_path

    def concat_file(self, file_path, file1, file2, output_path, output):
        """
        Concat file1 and file2 to be one output file 
        The i-th line of output = i-th line of file1 + '\t' + i-th line of file2
        file_path: the path to store file1 and file2
        output_path: the path to store output file
        """
        file1 = os.path.join(file_path, file1)
        file2 = os.path.join(file_path, file2)
        output = os.path.join(output_path, output)
        if not os.path.exists(output):
            os.system('paste ' + file1 + ' ' + file2 + ' > ' + output)

    def cat_file(self, dir_path, suffix, output_path, output):
        """
        Cat all the files in dir_path with suffix to be one output file 
        dir_path: the base directory to store input file
        suffix: suffix of file name
        output_path: the path to store output file
        """
        cmd = 'cat '
        file_list = os.listdir(dir_path)
        file_list.sort()
        for file in file_list:
            if file.endswith(suffix):
                cmd += os.path.join(dir_path, file) + ' '
        output = os.path.join(output_path, output)
        if not os.path.exists(output):
            os.system(cmd + '> ' + output)

    def build_dict(self, file_path, dict_path, dict_size=-1):
        """ 
        Create the dictionary for the file, Note that
        1. Valid characters include all printable characters
        2. There is distinction between uppercase and lowercase letters
        3. There is 3 special token: 
           <s>: the start of a sequence
           <e>: the end of a sequence
           <unk>: a word not included in dictionary
        file_path: the path to store file 
        dict_path: the path to store dictionary
        dict_size: word count of dictionary
                   if is -1, dictionary will contains all the words in file 
        """
        if not os.path.exists(dict_path):
            dictory = dict()
            with open(file_path, "r") as fdata:
                for line in fdata:
                    line = line.split('\t')
                    for line_split in line:
                        words = line_split.strip().split()
                        for word in words:
                            if word not in dictory:
                                dictory[word] = 1
                            else:
                                dictory[word] += 1
            output = open(dict_path, "w+")
            output.write('<s>\n<e>\n<unk>\n')
            count = 3
            for key, value in sorted(
                    dictory.items(), key=lambda d: d[1], reverse=True):
                output.write(key + "\n")
                count += 1
                if count == dict_size:
                    break
            self.dict_size = count

    def create_dataset(self,
                       dict_size=-1,
                       mergeDict=False,
                       suffixes=['.src', '.trg']):
        """
        Create seqToseq dataset 
        """
        # dataset_list and dir_list has one-to-one relationship
        train_dataset = os.path.join(self.data_path, self.train_dir_name)
        test_dataset = os.path.join(self.data_path, self.test_dir_name)
        gen_dataset = os.path.join(self.data_path, self.gen_dir_name)
        dataset_list = [train_dataset, test_dataset, gen_dataset]

        train_dir = os.path.join(self.output_path, self.train_dir_name)
        test_dir = os.path.join(self.output_path, self.test_dir_name)
        gen_dir = os.path.join(self.output_path, self.gen_dir_name)
        dir_list = [train_dir, test_dir, gen_dir]

        # create directory
        for dir in dir_list:
            if not os.path.exists(dir):
                os.mkdir(dir)

        # checkout dataset should be parallel corpora
        suffix_len = len(suffixes[0])
        for dataset in dataset_list:
            file_list = os.listdir(dataset)
            if len(file_list) % 2 == 1:
                raise RuntimeError("dataset should be parallel corpora")
            file_list.sort()
            for i in range(0, len(file_list), 2):
                if file_list[i][:-suffix_len] != file_list[i + 1][:-suffix_len]:
                    raise RuntimeError(
                        "source and target file name should be equal")

        # cat all the files with the same suffix in dataset
        for suffix in suffixes:
            for dataset in dataset_list:
                outname = os.path.basename(dataset) + suffix
                self.cat_file(dataset, suffix, dataset, outname)

        # concat parallel corpora and create file.list
        print 'concat parallel corpora for dataset'
        id = 0
        list = ['train.list', 'test.list', 'gen.list']
        for dataset in dataset_list:
            outname = os.path.basename(dataset)
            self.concat_file(dataset, outname + suffixes[0],
                             outname + suffixes[1], dir_list[id], outname)
            save_list([os.path.join(dir_list[id], outname)],
                      os.path.join(self.output_path, list[id]))
            id += 1

        # build dictionary for train data
        dict = ['src.dict', 'trg.dict']
        dict_path = [
            os.path.join(self.output_path, dict[0]),
            os.path.join(self.output_path, dict[1])
        ]
        if mergeDict:
            outname = os.path.join(train_dir, train_dataset.split('/')[-1])
            print 'build src dictionary for train data'
            self.build_dict(outname, dict_path[0], dict_size)
            print 'build trg dictionary for train data'
            os.system('cp ' + dict_path[0] + ' ' + dict_path[1])
        else:
            outname = os.path.join(train_dataset, self.train_dir_name)
            for id in range(0, 2):
                suffix = suffixes[id]
                print 'build ' + suffix[1:] + ' dictionary for train data'
                self.build_dict(outname + suffix, dict_path[id], dict_size)
        print 'dictionary size is', self.dict_size


def main():
    usage = "usage: \n" \
            "python %prog -i INPUT [-d DICTSIZE] [-m]"
    parser = OptionParser(usage)
    parser.add_option(
        "-i", action="store", dest="input", help="input original dataset path")
    parser.add_option(
        "-d",
        action="store",
        dest="dictsize",
        help="specified word count of dictionary")
    parser.add_option(
        "-m",
        "--mergeDict",
        action="store_true",
        dest="mergeDict",
        help="merge source and target dictionary")
    (options, args) = parser.parse_args()
    if options.input[-1] == os.path.sep:
        options.input = options.input[:-1]
    outname = os.path.basename(options.input)
    output_path = os.path.join(os.path.dirname(options.input), 'pre-' + outname)
    dictsize = int(options.dictsize) if options.dictsize else -1
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        data_creator = SeqToSeqDatasetCreater(options.input, output_path)
        data_creator.create_dataset(dictsize, options.mergeDict)


if __name__ == "__main__":
    main()
