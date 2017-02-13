import os, sys
import struct
import numpy as np
import paddle.v2 as paddle


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


def data(images, labels):
    for i in xrange(len(labels)):
        yield {"pixel": images[i, :], 'label': labels[i]}


def main():
    train_images, train_label = load_data('train')
    train_gen = data(train_images, train_label)
    train_data = paddle.data.CacheAllDataPool(train_gen, 128,
                                              ['pixel', 'label'])

    test_images, test_label = load_data('test')
    test_gen = data(test_images[0:128], test_label[0:128])
    test_data = paddle.data.CacheAllDataPool(test_gen, 128, ['pixel', 'label'],
                                             False)

    for data_batch in test_data:
        print data_batch


if __name__ == "__main__":
    main()
