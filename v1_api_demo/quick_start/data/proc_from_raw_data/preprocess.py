# -*- coding: UTF-8 -*-

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
1. Tokenize the words and punctuation 
2. pos sample : rating score 5; neg sample: rating score 1-2.

Usage:
    python preprocess.py -i data_file [random seed]
"""

import sys
import os
import operator
import gzip
from subprocess import Popen, PIPE
from optparse import OptionParser
import json
from multiprocessing import Queue
from multiprocessing import Pool
import multiprocessing

batch_size = 5000
word_count = {}
num_tokenize = max(1,
                   multiprocessing.cpu_count() - 2)  # parse + tokenize + save
max_queue_size = 8
parse_queue = Queue(maxsize=max_queue_size + num_tokenize)
tokenize_queue = Queue(maxsize=max_queue_size + num_tokenize)


def create_dict(data):
    """
    Create dictionary based on data, and saved in data_dir/dict.txt.
    The first line is unk \t -1.
    data: list, input data by batch.
    """
    for seq in data:
        try:
            for w in seq.lower().split():
                if w not in word_count:
                    word_count[w] = 1
                else:
                    word_count[w] += 1
        except:
            sys.stderr.write(seq + "\tERROR\n")


def parse(path):
    """
    Open .gz file.
    """
    sys.stderr.write(path)
    g = gzip.open(path, 'r')
    for l in g:
        yield json.loads(l)
    g.close()


def tokenize(sentences):
    """
    Use tokenizer.perl to tokenize input sentences.
    tokenizer.perl is tool of Moses.
    sentences : a list of input sentences.
    return: a list of processed text.
    """
    dir = './mosesdecoder-master/scripts/tokenizer/tokenizer.perl'
    if not os.path.exists(dir):
        sys.exit(
            "The ./mosesdecoder-master/scripts/tokenizer/tokenizer.perl does not exists."
        )
    tokenizer_cmd = [dir, '-l', 'en', '-q', '-']
    assert isinstance(sentences, list)
    text = "\n".join(sentences)
    tokenizer = Popen(tokenizer_cmd, stdin=PIPE, stdout=PIPE)
    tok_text, _ = tokenizer.communicate(text)
    toks = tok_text.split('\n')[:-1]
    return toks


def save_data(instance, data_dir, pre_fix, batch_num):
    """
    save data by batch
    """
    label = ['1' if pre_fix == 'pos' else '0' for i in range(len(instance))]
    lines = ['%s\t%s' % (label[i], instance[i]) for i in range(len(label))]
    file_name = os.path.join(data_dir, "%s_%s.txt" % (pre_fix, batch_num))
    file(file_name, 'w').write('\n'.join(lines) + '\n')


def tokenize_batch(id):
    """
    tokenize data by batch
    """
    while True:
        num_batch, instance, pre_fix = parse_queue.get()
        if num_batch == -1:  ### parse_queue finished
            tokenize_queue.put((-1, None, None))
            sys.stderr.write("Thread %s finish\n" % (id))
            break
        tokenize_instance = tokenize(instance)
        tokenize_queue.put((num_batch, tokenize_instance, pre_fix))
        sys.stderr.write('.')


def save_batch(data_dir, num_tokenize, data_dir_dict):
    """
        save data by batch
        build dict.txt
    """
    token_count = 0
    while True:
        num_batch, instance, pre_fix = tokenize_queue.get()
        if num_batch == -1:
            token_count += 1
            if token_count == num_tokenize:  #### tokenize finished.
                break
            else:
                continue
        save_data(instance, data_dir, pre_fix, num_batch)
        create_dict(instance)  ## update dict

    sys.stderr.write("save file finish\n")
    f = open(data_dir_dict, 'w')
    f.write('%s\t%s\n' % ('unk', '-1'))
    for k, v in sorted(word_count.items(), key=operator.itemgetter(1), \
                       reverse=True):
        f.write('%s\t%s\n' % (k, v))
    f.close()
    sys.stderr.write("build dict finish\n")


def parse_batch(data, num_tokenize):
    """
    parse data by batch
    parse -> tokenize -> save
    """
    raw_txt = parse(data)
    neg, pos = [], []
    count = 0
    sys.stderr.write("extract raw data\n")
    for l in raw_txt:
        rating = l["overall"]
        text = l["reviewText"].lower()  # # convert words to lower case
        if rating == 5.0 and text:
            pos.append(text)
        if rating < 3.0 and text:
            neg.append(text)
        if len(pos) == batch_size or len(neg) == batch_size:
            if len(pos) == batch_size:
                batch = pos
                pre_fix = 'pos'
            else:
                batch = neg
                pre_fix = 'neg'

            parse_queue.put((count, batch, pre_fix))
            count += 1
            if pre_fix == 'pos':
                pos = []
            else:
                neg = []

    if len(pos) > 0:
        parse_queue.put((count, pos, 'pos'))
        count += 1
    if len(neg) > 0:
        parse_queue.put((count, neg, 'neg'))
        count += 1
    for i in range(num_tokenize):
        parse_queue.put((-1, None, None))  #### for tokenize's input finished
    sys.stderr.write("parsing finish\n")


def option_parser():
    parser = OptionParser(usage="usage: python preprcoess.py "\
                                "-i data_path [options]")
    parser.add_option(
        "-i", "--data", action="store", dest="input", help="Input data path.")
    parser.add_option(
        "-s",
        "--seed",
        action="store",
        dest="seed",
        default=1024,
        help="Set random seed.")
    return parser.parse_args()


def main():
    reload(sys)
    sys.setdefaultencoding('utf-8')
    options, args = option_parser()
    data = options.input
    seed = options.seed
    data_dir_dict = os.path.join(os.path.dirname(data), 'dict.txt')
    data_dir = os.path.join(os.path.dirname(data), 'tmp')
    pool = Pool(processes=num_tokenize + 2)
    pool.apply_async(parse_batch, args=(data, num_tokenize))
    for i in range(num_tokenize):
        pool.apply_async(tokenize_batch, args=(str(i), ))
    pool.apply_async(save_batch, args=(data_dir, num_tokenize, data_dir_dict))
    pool.close()
    pool.join()

    file(os.path.join(os.path.dirname(data), 'labels.list'),
         'w').write('neg\t0\npos\t1\n')


if __name__ == '__main__':
    main()
