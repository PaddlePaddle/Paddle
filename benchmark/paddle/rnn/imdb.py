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

from __future__ import print_function
import six.moves.cPickle as pickle
import gzip
import os
import numpy


def get_dataset_file(dataset, default_dataset, origin):
    data_dir, data_file = os.path.split(dataset)
    if (not os.path.isfile(dataset)) and data_file == default_dataset:
        from six.moves import urllib
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    return dataset


def create_data(path="imdb.pkl"):

    if (not os.path.isfile('imdb.train.pkl')):
        path = get_dataset_file(
            path, "imdb.pkl",
            "http://www.iro.umontreal.ca/~lisa/deep/data/imdb.pkl")

        if path.endswith(".gz"):
            f = gzip.open(path, 'rb')
        else:
            f = open(path, 'rb')

        train_set = pickle.load(f)
        test_set = pickle.load(f)
        f.close()

        pickle.dump(train_set, open('imdb.train.pkl', 'wb'))
        pickle.dump(test_set, open('imdb.test.pkl', 'wb'))

    if (not os.path.isfile('train.list')):
        file('train.list', 'w').write('imdb.train.pkl\n')


def main():
    create_data('imdb.pkl')


if __name__ == "__main__":
    main()
