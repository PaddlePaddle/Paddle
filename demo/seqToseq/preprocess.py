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
from optparse import OptionParser

from paddle.v2.dataset.wmt14_util import SeqToSeqDatasetCreater


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
