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

__all__ = ['CIFAR10', 'CIFAR100', 'train_creator', 'test_creator']


def __download_file__(filename, url, md5):
    def __file_ok__():
        if not os.path.exists(filename):
            return False
        md5_hash = hashlib.md5()
        with open(filename, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)

        return md5_hash.hexdigest() == md5

    while not __file_ok__():
        response = urllib2.urlopen(url)
        with open(filename, mode='wb') as of:
            shutil.copyfileobj(fsrc=response, fdst=of)


def __read_one_batch__(batch):
    data = batch['data']
    labels = batch.get('labels', batch.get('fine_labels', None))
    assert labels is not None
    for sample, label in itertools.izip(data, labels):
        yield (sample / 255.0).astype(numpy.float32), int(label)


CIFAR10_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
CIFAR10_MD5 = 'c58f30108f718f92721af3b95e74349a'
CIFAR100_URL = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
CIFAR100_MD5 = 'eb9058c3a382ffc7106e4002c42a8d85'


class CIFAR(object):
    """
    CIFAR dataset reader. The base class for CIFAR-10 and CIFAR-100

    :param url: Download url.
    :param md5: File md5sum
    :param meta_filename: Meta file name in package.
    :param train_filename: Train file name in package.
    :param test_filename: Test file name in package.
    """

    def __init__(self, url, md5, meta_filename, train_filename, test_filename):
        filename = os.path.split(url)[-1]
        assert DATA_HOME is not None
        filepath = os.path.join(DATA_HOME, md5)
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        self.__full_file__ = os.path.join(filepath, filename)
        self.__meta_filename__ = meta_filename
        self.__train_filename__ = train_filename
        self.__test_filename__ = test_filename
        __download_file__(filename=self.__full_file__, url=url, md5=md5)

    def labels(self):
        """
        labels get all dataset label in order.
        :return: a list of label.
        :rtype: list[string]
        """
        with tarfile.open(self.__full_file__, mode='r') as f:
            name = [
                each_item.name for each_item in f
                if self.__meta_filename__ in each_item.name
            ][0]
            meta_f = f.extractfile(name)
            meta = cPickle.load(meta_f)
        for key in meta:
            if 'label' in key:
                return meta[key]
        else:
            raise RuntimeError("Unexpected branch.")

    def train(self):
        """
        Train Reader
        """
        return self.__read_batch__(self.__train_filename__)

    def test(self):
        """
        Test Reader
        """
        return self.__read_batch__(self.__test_filename__)

    def __read_batch__(self, sub_name):
        with tarfile.open(self.__full_file__, mode='r') as f:
            names = (each_item.name for each_item in f
                     if sub_name in each_item.name)

            for name in names:
                batch = cPickle.load(f.extractfile(name))
                for item in __read_one_batch__(batch):
                    yield item


class CIFAR10(CIFAR):
    """
    CIFAR-10 dataset, images are classified in 10 classes.
    """

    def __init__(self):
        super(CIFAR10, self).__init__(
            CIFAR10_URL,
            CIFAR10_MD5,
            meta_filename='batches.meta',
            train_filename='data_batch',
            test_filename='test_batch')


class CIFAR100(CIFAR):
    """
    CIFAR-100 dataset, images are classified in 100 classes.
    """

    def __init__(self):
        super(CIFAR100, self).__init__(
            CIFAR100_URL,
            CIFAR100_MD5,
            meta_filename='meta',
            train_filename='train',
            test_filename='test')


def train_creator():
    """
    Default train reader creator. Use CIFAR-10 dataset.
    """
    cifar = CIFAR10()
    return cifar.train


def test_creator():
    """
    Default test reader creator. Use CIFAR-10 dataset.
    """
    cifar = CIFAR10()
    return cifar.test


def unittest(label_count=100):
    cifar = globals()["CIFAR%d" % label_count]()
    assert len(cifar.labels()) == label_count
    for _ in cifar.test():
        pass
    for _ in cifar.train():
        pass


if __name__ == '__main__':
    unittest(10)
    unittest(100)
