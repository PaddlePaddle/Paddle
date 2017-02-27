"""
CIFAR Dataset.

URL: https://www.cs.toronto.edu/~kriz/cifar.html

the default train_creator, test_creator used for CIFAR-10 dataset.
"""
from config import DATA_HOME
import os
import hashlib
import urllib2
import shutil
import tarfile
import cPickle
import itertools
import numpy

__all__ = [
    'cifar_100_train_creator', 'cifar_100_test_creator', 'train_creator',
    'test_creator'
]

CIFAR10_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
CIFAR10_MD5 = 'c58f30108f718f92721af3b95e74349a'
CIFAR100_URL = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
CIFAR100_MD5 = 'eb9058c3a382ffc7106e4002c42a8d85'


def __read_batch__(filename, sub_name):
    def reader():
        def __read_one_batch_impl__(batch):
            data = batch['data']
            labels = batch.get('labels', batch.get('fine_labels', None))
            assert labels is not None
            for sample, label in itertools.izip(data, labels):
                yield (sample / 255.0).astype(numpy.float32), int(label)

        with tarfile.open(filename, mode='r') as f:
            names = (each_item.name for each_item in f
                     if sub_name in each_item.name)

            for name in names:
                batch = cPickle.load(f.extractfile(name))
                for item in __read_one_batch_impl__(batch):
                    yield item

    return reader


def download(url, md5):
    filename = os.path.split(url)[-1]
    assert DATA_HOME is not None
    filepath = os.path.join(DATA_HOME, md5)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    __full_file__ = os.path.join(filepath, filename)

    def __file_ok__():
        if not os.path.exists(__full_file__):
            return False
        md5_hash = hashlib.md5()
        with open(__full_file__, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)

        return md5_hash.hexdigest() == md5

    while not __file_ok__():
        response = urllib2.urlopen(url)
        with open(__full_file__, mode='wb') as of:
            shutil.copyfileobj(fsrc=response, fdst=of)
    return __full_file__


def cifar_100_train_creator():
    fn = download(url=CIFAR100_URL, md5=CIFAR100_MD5)
    return __read_batch__(fn, 'train')


def cifar_100_test_creator():
    fn = download(url=CIFAR100_URL, md5=CIFAR100_MD5)
    return __read_batch__(fn, 'test')


def train_creator():
    """
    Default train reader creator. Use CIFAR-10 dataset.
    """
    fn = download(url=CIFAR10_URL, md5=CIFAR10_MD5)
    return __read_batch__(fn, 'data_batch')


def test_creator():
    """
    Default test reader creator. Use CIFAR-10 dataset.
    """
    fn = download(url=CIFAR10_URL, md5=CIFAR10_MD5)
    return __read_batch__(fn, 'test_batch')


def unittest():
    for _ in train_creator()():
        pass
    for _ in test_creator()():
        pass


if __name__ == '__main__':
    unittest()
