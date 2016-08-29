# Copyright (c) 2016 Baidu, Inc. All Rights Reserved
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


def extract_dict_features(pair_file, feature_file, src_dict_file,
                          tgt_dict_file):
    src_dict = set()
    tgt_dict = set()

    with open(pair_file) as fin, open(feature_file, 'w') as feature_out, open(
            src_dict_file, 'w') as src_dict_out, open(tgt_dict_file,
                                                      'w') as tgt_dict_out:
        for line in fin:
            sentence, labels = line.strip().split('\t')
            sentence_list = sentence.split()
            labels_list = labels.split()

            src_dict.update(sentence_list)
            tgt_dict.update(labels_list)

            verb_index = labels_list.index('B-V')
            verb_feature = sentence_list[verb_index]

            mark = [0] * len(labels_list)
            if verb_index > 0:
                mark[verb_index - 1] = 1
                ctx_n1 = sentence_list[verb_index - 1]
            else:
                ctx_n1 = 'bos'
            ctx_n1_feature = ctx_n1

            mark[verb_index] = 1
            ctx_0_feature = sentence_list[verb_index]

            if verb_index < len(labels_list) - 2:
                mark[verb_index + 1] = 1
                ctx_p1 = sentence_list[verb_index + 1]
            else:
                ctx_p1 = 'eos'
            ctx_p1_feature = ctx_p1

            feature_str  = sentence + '\t' \
                           + verb_feature + '\t' \
                           + ctx_n1_feature + '\t' \
                           + ctx_0_feature + '\t' \
                           + ctx_p1_feature + '\t' \
                           + ' '.join([str(i) for i in mark]) + '\t' \
                           + labels

            feature_out.write(feature_str + '\n')

        src_dict_out.write('<unk>\n')
        src_dict_out.write('\n'.join(list(src_dict)))

        tgt_dict_out.write('\n'.join(list(tgt_dict)))


if __name__ == '__main__':

    usage = '-p pair_file -f feature_file -s source dictionary -t target dictionary '
    parser = OptionParser(usage)
    parser.add_option('-p', dest='pair_file', help='the pair file')
    parser.add_option(
        '-f', dest='feature_file', help='the file to store feature')
    parser.add_option(
        '-s', dest='src_dict', help='the file to store source dictionary')
    parser.add_option(
        '-t', dest='tgt_dict', help='the file to store target dictionary')

    (options, args) = parser.parse_args()

    extract_dict_features(options.pair_file, options.feature_file,
                          options.src_dict, options.tgt_dict)
