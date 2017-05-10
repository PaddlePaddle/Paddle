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


def read_labels(props_file):
    '''
    a sentence maybe has more than one verb, each verb has its label sequence
    label[],  is a 3-dimension list. 
    the first dim is to store all sentence's label seqs, len is the sentence number
    the second dim is to store all label sequences for one sentences
    the third dim is to store each label for one word
    '''
    labels = []
    with open(props_file) as fin:
        label_seqs_for_one_sentences = []
        one_seg_in_file = []
        for line in fin:
            line = line.strip()
            if line == '':
                for i in xrange(len(one_seg_in_file[0])):
                    a_kind_lable = [x[i] for x in one_seg_in_file]
                    label_seqs_for_one_sentences.append(a_kind_lable)
                labels.append(label_seqs_for_one_sentences)
                one_seg_in_file = []
                label_seqs_for_one_sentences = []
            else:
                part = line.split()
                one_seg_in_file.append(part)
    return labels


def read_sentences(words_file):
    sentences = []
    with open(words_file) as fin:
        s = ''
        for line in fin:
            line = line.strip()
            if line == '':
                sentences.append(s)
                s = ''
            else:
                s += line + ' '
    return sentences


def transform_labels(sentences, labels):
    sen_lab_pair = []
    for i in xrange(len(sentences)):
        if len(labels[i]) == 1:
            continue
        else:
            verb_list = []
            for x in labels[i][0]:
                if x != '-':
                    verb_list.append(x)

            for j in xrange(1, len(labels[i])):
                label_list = labels[i][j]
                current_tag = 'O'
                is_in_bracket = False
                label_seq = []
                verb_word = ''
                for ll in label_list:
                    if ll == '*' and is_in_bracket == False:
                        label_seq.append('O')
                    elif ll == '*' and is_in_bracket == True:
                        label_seq.append('I-' + current_tag)
                    elif ll == '*)':
                        label_seq.append('I-' + current_tag)
                        is_in_bracket = False
                    elif ll.find('(') != -1 and ll.find(')') != -1:
                        current_tag = ll[1:ll.find('*')]
                        label_seq.append('B-' + current_tag)
                        is_in_bracket = False
                    elif ll.find('(') != -1 and ll.find(')') == -1:
                        current_tag = ll[1:ll.find('*')]
                        label_seq.append('B-' + current_tag)
                        is_in_bracket = True
                    else:
                        print 'error:', ll
                sen_lab_pair.append((sentences[i], verb_list[j - 1], label_seq))
    return sen_lab_pair


def write_file(sen_lab_pair, output_file):
    with open(output_file, 'w') as fout:
        for x in sen_lab_pair:
            sentence = x[0]
            label_seq = ' '.join(x[2])
            assert len(sentence.split()) == len(x[2])
            fout.write(sentence + '\t' + x[1] + '\t' + label_seq + '\n')


if __name__ == '__main__':

    usage = '-w words_file -p props_file -o output_file'
    parser = OptionParser(usage)
    parser.add_option('-w', dest='words_file', help='the words file')
    parser.add_option('-p', dest='props_file', help='the props file')
    parser.add_option('-o', dest='output_file', help='the output_file')
    (options, args) = parser.parse_args()

    sentences = read_sentences(options.words_file)
    labels = read_labels(options.props_file)
    sen_lab_pair = transform_labels(sentences, labels)

    write_file(sen_lab_pair, options.output_file)
