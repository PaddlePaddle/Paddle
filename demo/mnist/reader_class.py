import os, sys
import struct
import numpy as np
from paddle.v2.data import IDataIter


def load_data(filename, dir='./data/raw_data/'):
    image = '-images-idx3-ubyte'
    label = '-labels-idx1-ubyte'
    if filename is 'train':
        image_file = os.path.join(dir, filename + image)
        label_file = os.path.join(dir, filename + label)
    else:
        image_file = os.path.join(dir, 't10k' + image)
        label_file = os.path.join(dir, 't10k' + label)

    with open(image_file, "rb") as f:
        num_magic, n, num_row, num_col = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, 'ubyte', count=n * num_row * num_col).\
            reshape(n, num_row * num_col).astype('float32')
        images = images / 255.0 * 2.0 - 1.0

    with open(label_file, "rb") as fn:
        num_magic, num_label = struct.unpack(">II", fn.read(8))
        labels = np.fromfile(fn, 'ubyte', count=num_label).astype('int32')

    return images, labels


class MNISTReader(IDataIter):
    def __init__(self, data, labels, batch_size, is_shuffle=False):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.num_examples = data.shape[0]
        self.is_shuffle = is_shuffle
        self.index_in_epoch = 0

    def __iter__(self):
        def shuffle():
            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)
            self.data = self.data[perm]
            self.labels = self.labels[perm]

        if self.is_shuffle:
            shuffle()
        return self

    def next(self):
        if self.index_in_epoch >= self.num_examples:
            self.index_in_epoch = 0
            raise StopIteration

        start = self.index_in_epoch
        self.index_in_epoch += self.batch_size
        end = min(self.index_in_epoch, self.num_examples)
        return {'pixel': self.data[start:end], 'label': self.labels[start:end]}


def main():
    train_images, train_label = load_data('train')
    train_data = MNISTReader(train_images, train_label, 128, True)

    test_images, test_label = load_data('test')
    test_data = MNISTReader(test_images, test_label, 128, False)

    for i in xrange(2):
        print '---start pass---'
        for data_batch in test_data:
            print data_batch['pixel'].shape


if __name__ == "__main__":
    main()
