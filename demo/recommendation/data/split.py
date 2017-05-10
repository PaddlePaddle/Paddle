#!/bin/env python2
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
Separate movielens 1m dataset to train/test file.

Usage:
    ./separate.py <input_file> [--test_ratio=<test_ratio>] [--delimiter=<delimiter>]
    ./separate.py -h | --help

Options:
    -h --help                       Show this screen.
    --version                       Show version.
    --test_ratio=<test_ratio>       Test ratio for separate [default: 0.1].
    --delimiter=<delimiter>         File delimiter [default: ,].
"""
import docopt
import collections
import random


def process(test_ratio, input_file, delimiter, **kwargs):
    test_ratio = float(test_ratio)
    rating_dict = collections.defaultdict(list)
    with open(input_file, 'r') as f:
        for line in f:
            user_id = int(line.split(delimiter)[0])
            rating_dict[user_id].append(line.strip())

    with open(input_file + ".train", 'w') as train_file:
        with open(input_file + ".test", 'w') as test_file:
            for k in rating_dict.keys():
                lines = rating_dict[k]
                assert isinstance(lines, list)
                random.shuffle(lines)
                test_len = int(len(lines) * test_ratio)
                for line in lines[:test_len]:
                    print >> test_file, line

                for line in lines[test_len:]:
                    print >> train_file, line


if __name__ == '__main__':
    args = docopt.docopt(__doc__, version='0.1.0')
    kwargs = dict()
    for key in args.keys():
        if key != '--help':
            param_name = key
            assert isinstance(param_name, str)
            param_name = param_name.replace('<', '')
            param_name = param_name.replace('>', '')
            param_name = param_name.replace('--', '')
            kwargs[param_name] = args[key]
    process(**kwargs)
