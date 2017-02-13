import os, sys
import struct
import numpy as np


class DataReader(object):
    def __init__(self, data, labels, batch_size, is_shuffle=False):
        assert data.shape[0] == labels.shape[0], (
            'data.shape: %s labels.shape: %s' % (data.shape, labels.shape))
        self.num_examples = data.shape[0]
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.is_shuffle = is_shuffle
        self.index_in_epoch = 0
        if self.is_shuffle:
            self.shuffle()

    def __iter__(self):
        return self

    def shuffle(self):
        perm = np.arange(self.num_examples)
        np.random.shuffle(perm)
        self.data = self.data[perm]
        self.labels = self.labels[perm]

    def next(self):
        start = self.index_in_epoch
        self.index_in_epoch += self.batch_size
        end = min(self.index_in_epoch, self.num_examples)
        if self.index_in_epoch >= self.num_examples:
            self.index_in_epoch = 0
            if self.is_shuffle:
                self.shuffle()
            raise StopIteration
        return {'pixel': self.data[start:end], 'label': self.labels[start:end]}


def create_datasets(dir='./data/raw_data/'):
    '''
    数据download 和 load可以依据https://github.com/PaddlePaddle/Paddle/pull/872来简化
    '''

    def load_data(filename, dir):
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
            labels = np.fromfile(fn, 'ubyte', count=num_label).astype('int')

        return images, labels

    train_image, train_label = load_data('train', dir)
    test_image, test_label = load_data('test', dir)

    trainset = DataReader(train_image, train_label, 128, True)
    testset = DataReader(test_image, test_label, 128, False)
    return trainset, testset


def main():
    train_data, test_data = create_datasets()
    for data_batch in test_data:
        print data_batch


if __name__ == "__main__":
    main()
