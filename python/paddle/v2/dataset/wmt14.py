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
import os
import os.path
import tarfile

import paddle.v2.dataset.common
from wmt14_util import SeqToSeqDatasetCreater

__all__ = ['train', 'test', 'build_dict']

URL_DEV_TEST = 'http://www-lium.univ-lemans.fr/~schwenk/cslm_joint_paper/data/dev+test.tgz'
MD5_DEV_TEST = '7d7897317ddd8ba0ae5c5fa7248d3ff5'
# this is a small set of data for test. The original data is too large and will be add later.
URL_TRAIN = 'http://paddlepaddle.bj.bcebos.com/demo/wmt_shrinked_data/wmt14.tgz'
MD5_TRAIN = '7373473f86016f1f48037c9c340a2d5b'

START = "<s>"
END = "<e>"
UNK = "<unk>"
UNK_IDX = 2

DEFAULT_DATA_DIR = "./data"
ORIGIN_DATA_DIR = "wmt14"
INNER_DATA_DIR = "pre-wmt14"
SRC_DICT = INNER_DATA_DIR + "/src.dict"
TRG_DICT = INNER_DATA_DIR + "/trg.dict"
TRAIN_FILE = INNER_DATA_DIR + "/train/train"


def __process_data__(data_path, dict_size=None):
    downloaded_data = os.path.join(data_path, ORIGIN_DATA_DIR)
    if not os.path.exists(downloaded_data):
        # 1. download and extract tgz.
        with tarfile.open(
                paddle.v2.dataset.common.download(URL_TRAIN, 'wmt14',
                                                  MD5_TRAIN)) as tf:
            tf.extractall(data_path)

    # 2. process data file to intermediate format.
    processed_data = os.path.join(data_path, INNER_DATA_DIR)
    if not os.path.exists(processed_data):
        dict_size = dict_size or -1
        data_creator = SeqToSeqDatasetCreater(downloaded_data, processed_data)
        data_creator.create_dataset(dict_size, mergeDict=False)


def __read_to_dict__(dict_path, count):
    with open(dict_path, "r") as fin:
        out_dict = dict()
        for line_count, line in enumerate(fin):
            if line_count <= count:
                out_dict[line.strip()] = line_count
            else:
                break
    return out_dict


def __reader__(file_name, src_dict, trg_dict):
    with open(file_name, 'r') as f:
        for line_count, line in enumerate(f):
            line_split = line.strip().split('\t')
            if len(line_split) != 2:
                continue
            src_seq = line_split[0]  # one source sequence
            src_words = src_seq.split()
            src_ids = [
                src_dict.get(w, UNK_IDX) for w in [START] + src_words + [END]
            ]

            trg_seq = line_split[1]  # one target sequence
            trg_words = trg_seq.split()
            trg_ids = [trg_dict.get(w, UNK_IDX) for w in trg_words]

            # remove sequence whose length > 80 in training mode
            if len(src_ids) > 80 or len(trg_ids) > 80:
                continue
            trg_ids_next = trg_ids + [trg_dict[END]]
            trg_ids = [trg_dict[START]] + trg_ids

            yield src_ids, trg_ids, trg_ids_next


def train(data_dir=None, dict_size=None):
    data_dir = data_dir or DEFAULT_DATA_DIR
    __process_data__(data_dir, dict_size)
    src_lang_dict = os.path.join(data_dir, SRC_DICT)
    trg_lang_dict = os.path.join(data_dir, TRG_DICT)
    train_file_name = os.path.join(data_dir, TRAIN_FILE)

    default_dict_size = len(open(src_lang_dict, "r").readlines())

    if dict_size > default_dict_size:
        raise ValueError("dict_dim should not be larger then the "
                         "length of word dict")

    real_dict_dim = dict_size or default_dict_size

    src_dict = __read_to_dict__(src_lang_dict, real_dict_dim)
    trg_dict = __read_to_dict__(trg_lang_dict, real_dict_dim)

    return lambda: __reader__(train_file_name, src_dict, trg_dict)
