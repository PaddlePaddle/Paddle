# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
from hapi.text.senta.data_reader import load_vocab
from hapi.text.senta.data_reader import data_reader
from paddle.io import DataLoader


class SentaProcessor(object):
    def __init__(self, data_dir, vocab_path, random_seed=None):
        self.data_dir = data_dir
        self.vocab = load_vocab(vocab_path)
        self.num_examples = {"train": -1, "dev": -1, "infer": -1}
        np.random.seed(random_seed)

    def get_train_examples(self, data_dir, epoch, shuffle, batch_size, places, padding_size):
        train_reader = data_reader((self.data_dir + "/train.tsv"), self.vocab,
                           self.num_examples, "train", epoch, padding_size, shuffle)
        loader = DataLoader.from_generator(capacity=50, return_list=True)
        loader.set_sample_generator(train_reader, batch_size=batch_size, drop_last=False, places=places)
        return loader
        

    def get_dev_examples(self, data_dir, epoch, shuffle, batch_size, places, padding_size):
        dev_reader = data_reader((self.data_dir + "/dev.tsv"), self.vocab,
                           self.num_examples, "dev", epoch, padding_size, shuffle)
        loader = DataLoader.from_generator(capacity=50, return_list=True)
        loader.set_sample_generator(dev_reader, batch_size=batch_size, drop_last=False, places=places)
        return loader

    def get_test_examples(self, data_dir, epoch, batch_size, places, padding_size):
        test_reader = data_reader((self.data_dir + "/test.tsv"), self.vocab,
                           self.num_examples, "infer", epoch, padding_size)
        loader = DataLoader.from_generator(capacity=50, return_list=True)
        loader.set_sample_generator(test_reader, batch_size=batch_size, drop_last=False, places=places)
        return loader

    def get_labels(self):
        return ["0", "1"]

    def get_num_examples(self, phase):
        if phase not in ['train', 'dev', 'infer']:
            raise ValueError(
                "Unknown phase, which should be in ['train', 'dev', 'infer'].")
        return self.num_examples[phase]

    def get_train_progress(self):
        return self.current_train_example, self.current_train_epoch

    def data_generator(self, padding_size, batch_size, places, phase='train', epoch=1, shuffle=True):
        if phase == "train":
            return self.get_train_examples(self.data_dir, epoch, shuffle, batch_size, places, padding_size)
        elif phase == "dev":
            return self.get_dev_examples(self.data_dir, epoch, shuffle, batch_size, places, padding_size)
        elif phase == "infer":
            return self.get_test_examples(self.data_dir, epoch, batch_size, places, padding_size)
        else:
            raise ValueError(
                "Unknown phase, which should be in ['train', 'dev', 'infer'].")
