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

'''
1. remove HTML before tokensizing 
2. pos sample : rating score 5; neg sample: rating score 1-2.
3. size of pos : neg = 1:1.
4. size of testing set = min(25k, len(all_data) * 0.1), others is traning set.
5. distinct train set and test set.

Usage:
    python preprocess.py -i data_file [random seed]
'''

import sys,os
import re
import operator
import gzip,math
import random
import numpy as np
from bs4 import BeautifulSoup
from subprocess import Popen, PIPE
from optparse import OptionParser

def parse(path):
    """
    Open .gz file.
    """
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

def clean(review):
    """
    Clean input review: remove HTML, convert words to lower cases.
    """
    # Remove HTML
    review_text = BeautifulSoup(review, "html.parser").get_text()

    # Convert words to lower case
    review_text = review_text.lower()
    return review_text

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

def create_dict(data, data_dir):
    """
    Create dictionary based on data, and saved in data_dir/dict.txt.
    The first line is unk \t -1. 
    data: list, input data.
    data_dir: path to save dict.
    """
    word_count = {}
    for seq in data:
        try:
            for w in seq.lower().split():
                if w not in word_count:
                    word_count[w] = 1
                else:
                    word_count[w] += 1
        except:
            sys.stderr.write(seq+"\tERROR\n")
    f = open(os.path.join(data_dir, 'dict.txt'), 'w')
    f.write('%s\t%s\n' % ('unk', '-1'))
    for k, v in sorted(word_count.items(), key=operator.itemgetter(1),\
                      reverse=True):
        f.write('%s\t%s\n' % (k, v))
    f.close()

def save_data(data, data_dir, prefix = ""):
    file_name = os.path.join(data_dir, "%s.txt" % (prefix))
    file(file_name,'w').write('\n'.join(data)+'\n')
    file(os.path.join(data_dir, prefix+'.list'),'w').write('%s\n' % file_name)

def split_data(raw_txt):
    """
    Extract positive and negative sample.
    """
    pos = []
    neg = []
    count = 0
    dup_cnt = 0
    sys.stderr.write("extract raw data")
    for l in raw_txt:
        rating = l["overall"]
        text = clean(l["reviewText"])
        if rating == 5.0 and text:
            pos.append(text)
        if rating < 3.0 and text:
            neg.append(text)
        count += 1
        if count % 20000==0:
            sys.stderr.write(".")
    sys.stderr.write("\n")
    return pos, neg

def preprocess(pos_in, neg_in, data_dir, rand_seed):
    # tokenize
    sys.stderr.write("tokenize...\n")
    tmppos = tokenize(pos_in)
    tmpneg = tokenize(neg_in)
    cnt = len(tmppos) + len(tmpneg)

    # unique smaples
    tmppos = list(set(tmppos))
    tmpneg = list(set(tmpneg))
    dup_cnt = cnt - len(tmppos) - len(tmpneg)
    sys.stderr.write("\ntotal size of data set: %d, duplicate data: %d\n" % (cnt, dup_cnt))

    # keep same size of positive and negative sample
    min_len = min(len(tmppos), len(tmpneg))
    tmppos = tmppos[0:min_len]
    tmpneg = tmpneg[0:min_len]

    # creat dictionary
    sys.stderr.write("create dict with train and test data...\n")
    all_data = tmppos + tmpneg
    create_dict(all_data, data_dir)

    # split into train set and test set
    sys.stderr.write("split data...\n")
    pos = ["1\t"+i for i in tmppos]
    neg = ["0\t"+i for i in tmpneg]
    random.seed(rand_seed)
    random.shuffle(pos)
    random.shuffle(neg)

    # split into test set and train set
    test_len = min(12500, int(min_len * 0.1))
    test = pos[0:test_len] + neg[0:test_len]
    train = pos[test_len:] + neg[test_len:]

    # save data
    sys.stderr.write("save data...\n")
    save_data(train, data_dir, prefix = 'train')
    save_data(test, data_dir, prefix = 'test')
    file(os.path.join(data_dir,'labels.list'),'w').write('neg\t0\npos\t1\n')

def option_parser():
    parser = OptionParser(usage="usage: python preprcoess.py "\
                                "-i data_path [options]")
    parser.add_option("-i", "--data", action="store",
                      dest="input", help="Input data path.")
    parser.add_option("-s", "--seed", action="store",
                      dest="seed", default=1024,
                      help="Set random seed.")
    return parser.parse_args()

def main():
    reload(sys)
    sys.setdefaultencoding('utf-8')
    options, args = option_parser()
    data=options.input
    seed=options.seed
    data_dir = os.path.dirname(data)
    pos, neg = split_data(parse(data))
    preprocess(pos, neg, data_dir, seed)
    sys.stderr.write("Done.\n")

if __name__ == '__main__':
    main()
