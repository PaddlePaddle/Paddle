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
    python extract_para.py --preModel PREMODEL --preDict PREDICT \
                            --usrModel USRMODEL --usrDict USRDICT -d DIM

Options:
    -h, --help          show this help message and exit
    --preModel PREMODEL the name of pretrained embedding model
    --preDict PREDICT   the name of pretrained dictionary
    --usrModel usrModel the name of output usr embedding model
    --usrDict usrDict   the name of user specified dictionary
    -d DIM              dimension of parameter
"""
from optparse import OptionParser
import struct


def get_row_index(preDict, usrDict):
    """
    Get the row positions for all words in user dictionary from pre-trained dictionary.
    return: a list of row positions
    Example: preDict='a\nb\nc\n', usrDict='a\nc\n', then return [0,2]
    """
    pos = []
    index = dict()
    with open(preDict, "r") as f:
        for line_index, line in enumerate(f):
            word = line.strip().split()[0]
            index[word] = line_index
    with open(usrDict, "r") as f:
        for line in f:
            word = line.strip().split()[0]
            pos.append(index[word])
    return pos


def extract_parameters_by_usrDict(preModel, preDict, usrModel, usrDict,
                                  paraDim):
    """
    Extract desired parameters from a pretrained embedding model based on user dictionary
    """
    if paraDim not in [32, 64, 128, 256]:
        raise RuntimeError("We only support 32, 64, 128, 256 dimensions now")

    fi = open(preModel, "rb")
    fo = open(usrModel, "wb")

    # write filehead
    rowIndex = get_row_index(preDict, usrDict)
    newHead = struct.pack("iil", 0, 4, len(rowIndex) * paraDim)
    fo.write(newHead)
    bytes = 4 * paraDim
    for i in range(0, len(rowIndex)):
        # find the absolute position of input file
        fi.seek(rowIndex[i] * bytes + 16, 0)
        fo.write(fi.read(bytes))

    print "extract parameters finish, total", len(rowIndex), "lines"
    fi.close()


def main():
    """
    Main entry for running paraconvert.py 
    """
    usage = "usage: \n" \
            "python %prog --preModel PREMODEL --preDict PREDICT" \
            " --usrModel USRMODEL --usrDict USRDICT -d DIM"
    parser = OptionParser(usage)
    parser.add_option(
        "--preModel",
        action="store",
        dest="preModel",
        help="the name of pretrained embedding model")
    parser.add_option(
        "--preDict",
        action="store",
        dest="preDict",
        help="the name of pretrained dictionary")
    parser.add_option(
        "--usrModel",
        action="store",
        dest="usrModel",
        help="the name of output usr embedding model")
    parser.add_option(
        "--usrDict",
        action="store",
        dest="usrDict",
        help="the name of user specified dictionary")
    parser.add_option(
        "-d", action="store", dest="dim", help="dimension of parameter")
    (options, args) = parser.parse_args()
    extract_parameters_by_usrDict(options.preModel, options.preDict,
                                  options.usrModel, options.usrDict,
                                  int(options.dim))


if __name__ == '__main__':
    main()
