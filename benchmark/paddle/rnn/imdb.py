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
