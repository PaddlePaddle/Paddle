#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import os.path
import io
import numpy as np
import tensorflow as tf

# tflearn
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb

FLAGS = tf.app.flags.FLAGS


class DataSet(object):
    def __init__(self, data, labels):
        assert data.shape[0] == labels.shape[0], (
            'data.shape: %s labels.shape: %s' % (data.shape, labels.shape))
        self._num_examples = data.shape[0]

        self._data = data
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        assert batch_size <= self._num_examples

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._data = self._data[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size

        end = self._index_in_epoch

        return self._data[start:end], self._labels[start:end]


def create_datasets(file_path, vocab_size=30000, val_fraction=0.0):

    # IMDB Dataset loading
    train, test, _ = imdb.load_data(
        path=file_path,
        n_words=vocab_size,
        valid_portion=val_fraction,
        sort_by_len=False)
    trainX, trainY = train
    testX, testY = test

    # Data preprocessing
    # Sequence padding
    trainX = pad_sequences(trainX, maxlen=FLAGS.max_len, value=0.)
    testX = pad_sequences(testX, maxlen=FLAGS.max_len, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    train_dataset = DataSet(trainX, trainY)

    return train_dataset


def main():
    create_datasets('imdb.pkl')


if __name__ == "__main__":
    main()
