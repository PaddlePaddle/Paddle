import random
from paddle.v2.image import load_and_transform
import paddle.v2 as paddle
from multiprocessing import cpu_count


def train_mapper(sample):
    '''
    map image path to type needed by model input layer for the training set
    '''
    img, label = sample
    img = paddle.image.load_image(img)
    img = paddle.image.simple_transform(img, 256, 224, True)
    return img.flatten().astype('float32'), label


def test_mapper(sample):
    '''
    map image path to type needed by model input layer for the test set
    '''
    img, label = sample
    img = paddle.image.load_image(img)
    img = paddle.image.simple_transform(img, 256, 224, True)
    return img.flatten().astype('float32'), label


def train_reader(train_list, buffered_size=1024):
    def reader():
        with open(train_list, 'r') as f:
            lines = [line.strip() for line in f]
            for line in lines:
                img_path, lab = line.strip().split('\t')
                yield img_path, int(lab)

    return paddle.reader.xmap_readers(train_mapper, reader,
                                      cpu_count(), buffered_size)


def test_reader(test_list, buffered_size=1024):
    def reader():
        with open(test_list, 'r') as f:
            lines = [line.strip() for line in f]
            for line in lines:
                img_path, lab = line.strip().split('\t')
                yield img_path, int(lab)

    return paddle.reader.xmap_readers(test_mapper, reader,
                                      cpu_count(), buffered_size)


if __name__ == '__main__':
    #for im in train_reader('train.list'):
    #    print len(im[0])
    #for im in train_reader('test.list'):
    #    print len(im[0])
    paddle.dataset.flowers.train()
