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

import sys
import os
from optparse import OptionParser


def extract_dict_features(pair_file, feature_file):

    with open(pair_file) as fin, open(feature_file, 'w') as feature_out:
        for line in fin:
            sentence, predicate, labels = line.strip().split('\t')
            sentence_list = sentence.split()
            labels_list = labels.split()

            verb_index = labels_list.index('B-V')

            mark = [0] * len(labels_list)
            if verb_index > 0:
                mark[verb_index - 1] = 1
                ctx_n1 = sentence_list[verb_index - 1]
            else:
                ctx_n1 = 'bos'

            if verb_index > 1:
                mark[verb_index - 2] = 1
                ctx_n2 = sentence_list[verb_index - 2]
            else:
                ctx_n2 = 'bos'

            mark[verb_index] = 1
            ctx_0 = sentence_list[verb_index]

            if verb_index < len(labels_list) - 1:
                mark[verb_index + 1] = 1
                ctx_p1 = sentence_list[verb_index + 1]
            else:
                ctx_p1 = 'eos'

            if verb_index < len(labels_list) - 2:
                mark[verb_index + 2] = 1
                ctx_p2 = sentence_list[verb_index + 2]
            else:
                ctx_p2 = 'eos'


            feature_str  = sentence + '\t' \
                           + predicate + '\t' \
                           + ctx_n2 + '\t' \
                           + ctx_n1 + '\t' \
                           + ctx_0 + '\t' \
                           + ctx_p1 + '\t' \
                           + ctx_p2 + '\t' \
                           + ' '.join([str(i) for i in mark]) + '\t' \
                           + labels

            feature_out.write(feature_str + '\n')


if __name__ == '__main__':

    usage = '-p pair_file -f feature_file'
    parser = OptionParser(usage)
    parser.add_option('-p', dest='pair_file', help='the pair file')
    parser.add_option('-f', dest='feature_file', help='the feature file')

    (options, args) = parser.parse_args()

    extract_dict_features(options.pair_file, options.feature_file)
