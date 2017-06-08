import os, sys
import struct
import numpy as np


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


def reader(data, label, batch_size, is_shuffle=True):
    def make_shuffle(data, label):
        perm = np.arange(len(data))
        np.random.shuffle(perm)
        data = data[perm]
        label = label[perm]

    def make_minibatch(data, label):
        if is_shuffle:
            print is_shuffle
            make_shuffle(data, label)
        for start in xrange(0, len(data), batch_size):
            end = min(start + batch_size, len(data))
            # generate mini-batch, data
            yield {'pixel': data[start:end], 'label': label[start:end]}

    return make_minibatch(data, label)


def main():
    train_images, train_label = load_data('train')
    train_data = reader(train_images, train_label, 128, True)

    test_images, test_label = load_data('test')
    test_data = reader(test_images, test_label, 128, True)

    for i in xrange(2):
        print '---start pass---'
        for data_batch in gen:
            print data_batch['pixel'].shape


if __name__ == "__main__":
    main()
