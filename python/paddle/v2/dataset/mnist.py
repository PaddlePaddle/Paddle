"""
MNIST dataset.
"""
import numpy
import paddle.v2.dataset.common
import subprocess

__all__ = ['train', 'test']

URL_PREFIX = 'http://yann.lecun.com/exdb/mnist/'
TEST_IMAGE_URL = URL_PREFIX + 't10k-images-idx3-ubyte.gz'
TEST_IMAGE_MD5 = '25e3cc63507ef6e98d5dc541e8672bb6'
TEST_LABEL_URL = URL_PREFIX + 't10k-labels-idx1-ubyte.gz'
TEST_LABEL_MD5 = '4e9511fe019b2189026bd0421ba7b688'
TRAIN_IMAGE_URL = URL_PREFIX + 'train-images-idx3-ubyte.gz'
TRAIN_IMAGE_MD5 = 'f68b3c2dcbeaaa9fbdd348bbdeb94873'
TRAIN_LABEL_URL = URL_PREFIX + 'train-labels-idx1-ubyte.gz'
TRAIN_LABEL_MD5 = 'd53e105ee54ea40749a09fcbcd1e9432'


def reader_creator(image_filename, label_filename, buffer_size):
    def reader():
        # According to http://stackoverflow.com/a/38061619/724872, we
        # cannot use standard package gzip here.
        m = subprocess.Popen(["zcat", image_filename], stdout=subprocess.PIPE)
        m.stdout.read(16)  # skip some magic bytes

        l = subprocess.Popen(["zcat", label_filename], stdout=subprocess.PIPE)
        l.stdout.read(8)  # skip some magic bytes

        while True:
            labels = numpy.fromfile(
                l.stdout, 'ubyte', count=buffer_size).astype("int")

            if labels.size != buffer_size:
                break  # numpy.fromfile returns empty slice after EOF.

            images = numpy.fromfile(
                m.stdout, 'ubyte', count=buffer_size * 28 * 28).reshape(
                    (buffer_size, 28 * 28)).astype('float32')

            images = images / 255.0 * 2.0 - 1.0

            for i in xrange(buffer_size):
                yield images[i, :], int(labels[i])

        m.terminate()
        l.terminate()

    return reader


def train():
    return reader_creator(
        paddle.v2.dataset.common.download(TRAIN_IMAGE_URL, 'mnist',
                                          TRAIN_IMAGE_MD5),
        paddle.v2.dataset.common.download(TRAIN_LABEL_URL, 'mnist',
                                          TRAIN_LABEL_MD5), 100)


def test():
    return reader_creator(
        paddle.v2.dataset.common.download(TEST_IMAGE_URL, 'mnist',
                                          TEST_IMAGE_MD5),
        paddle.v2.dataset.common.download(TEST_LABEL_URL, 'mnist',
                                          TEST_LABEL_MD5), 100)
