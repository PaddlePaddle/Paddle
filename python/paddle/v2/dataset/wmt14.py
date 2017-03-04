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
wmt14 dataset
"""
import paddle.v2.dataset.common
import tarfile
import os.path
import itertools

__all__ = ['train', 'test', 'build_dict']

URL_DEV_TEST = 'http://www-lium.univ-lemans.fr/~schwenk/cslm_joint_paper/data/dev+test.tgz'
MD5_DEV_TEST = '7d7897317ddd8ba0ae5c5fa7248d3ff5'
URL_TRAIN = 'http://localhost:8000/train.tgz'
MD5_TRAIN = '72de99da2830ea5a3a2c4eb36092bbc7'


def word_count(f, word_freq=None):
    add = paddle.v2.dataset.common.dict_add
    if word_freq == None:
        word_freq = {}

    for l in f:
        for w in l.strip().split():
            add(word_freq, w)
        add(word_freq, '<s>')
        add(word_freq, '<e>')

    return word_freq


def get_word_dix(word_freq):
    TYPO_FREQ = 50
    word_freq = filter(lambda x: x[1] > TYPO_FREQ, word_freq.items())
    word_freq_sorted = sorted(word_freq, key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*word_freq_sorted))
    word_idx = dict(zip(words, xrange(len(words))))
    word_idx['<unk>'] = len(words)
    return word_idx


def get_word_freq(train, dev):
    word_freq = word_count(train, word_count(dev))
    if '<unk>' in word_freq:
        # remove <unk> for now, since we will set it as last index
        del word_freq['<unk>']
    return word_freq


def build_dict():
    base_dir = './wmt14-data'
    train_en_filename = base_dir + '/train/train.en'
    train_fr_filename = base_dir + '/train/train.fr'
    dev_en_filename = base_dir + '/dev/ntst1213.en'
    dev_fr_filename = base_dir + '/dev/ntst1213.fr'

    if not os.path.exists(train_en_filename) or not os.path.exists(
            train_fr_filename):
        with tarfile.open(
                paddle.v2.dataset.common.download(URL_TRAIN, 'wmt14',
                                                  MD5_TRAIN)) as tf:
            tf.extractall(base_dir)

    if not os.path.exists(dev_en_filename) or not os.path.exists(
            dev_fr_filename):
        with tarfile.open(
                paddle.v2.dataset.common.download(URL_DEV_TEST, 'wmt14',
                                                  MD5_DEV_TEST)) as tf:
            tf.extractall(base_dir)

    f_en = open(train_en_filename)
    f_fr = open(train_fr_filename)
    f_en_dev = open(dev_en_filename)
    f_fr_dev = open(dev_fr_filename)

    word_freq_en = get_word_freq(f_en, f_en_dev)
    word_freq_fr = get_word_freq(f_fr, f_fr_dev)

    f_en.close()
    f_fr.close()
    f_en_dev.close()
    f_fr_dev.close()

    return get_word_dix(word_freq_en), get_word_dix(word_freq_fr)


def reader_creator(directory, path_en, path_fr, URL, MD5, dict_en, dict_fr):
    def reader():
        if not os.path.exists(path_en) or not os.path.exists(path_fr):
            with tarfile.open(
                    paddle.v2.dataset.common.download(URL, 'wmt14', MD5)) as tf:
                tf.extractall(directory)

        f_en = open(path_en)
        f_fr = open(path_fr)
        UNK_en = dict_en['<unk>']
        UNK_fr = dict_fr['<unk>']

        for en, fr in itertools.izip(f_en, f_fr):
            src_ids = [dict_en.get(w, UNK_en) for w in en.strip().split()]
            tar_ids = [
                dict_fr.get(w, UNK_fr)
                for w in ['<s>'] + fr.strip().split() + ['<e>']
            ]

            # remove sequence whose length > 80 in training mode
            if len(src_ids) == 0 or len(tar_ids) <= 1 or len(
                    src_ids) > 80 or len(tar_ids) > 80:
                continue

            yield src_ids, tar_ids[:-1], tar_ids[1:]

        f_en.close()
        f_fr.close()

    return reader


def train(dict_en, dict_fr):
    directory = './wmt14-data'
    return reader_creator(directory, directory + '/train/train.en',
                          directory + '/train/train.fr', URL_TRAIN, MD5_TRAIN,
                          dict_en, dict_fr)


def test(dict_en, dict_fr):
    directory = './wmt14-data'
    return reader_creator(directory, directory + '/dev/ntst1213.en',
                          directory + '/dev/ntst1213.fr', URL_DEV_TEST,
                          MD5_DEV_TEST, dict_en, dict_fr)
