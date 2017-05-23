# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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
"""
CIFAR dataset.

This module will download dataset from
http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html 
and parse train/test set intopaddle reader creators.

This set contains images of flowers belonging to 102 different categories. 
The images were acquired by searching the web and taking pictures. There are a
minimum of 40 images for each category.

The database was used in:

Nilsback, M-E. and Zisserman, A. Automated flower classification over a large
 number of classes.Proceedings of the Indian Conference on Computer Vision, 
Graphics and Image Processing (2008) 
http://www.robots.ox.ac.uk/~vgg/publications/papers/nilsback08.{pdf,ps.gz}.

"""
import cPickle
import itertools
from common import download
import tarfile
import scipy.io as scio
from image import *
import os
from multiprocessing import Process
from multiprocessing import Pool
from multiprocessing import cpu_count
import numpy as np
import paddle.v2 as paddle
__all__ = ['train', 'test', 'valid']

DATA_URL = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz'
LABEL_URL = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat'
SETID_URL = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat'
DATA_MD5 = '52808999861908f626f3c1f4e79d11fa'
LABEL_MD5 = 'e0620be6f572b9609742df49c70aed4d'
SETID_MD5 = 'a5357ecc9cb78c4bef273ce3793fc85c'


def extract_file(tarFile):
    '''
    Extract tar file to tmp dir.
    
    Example usage:

    .. code-block:: python
        tmp = extract_file("/home/work/test.tar.gz")

    :param tarFile: target tar file
    :type tarFile: string
    :return: extracted dir. For example: 
            '/home/work/test/' while input is '/home/work/test.tar.gz'
    :rtype: string
    '''
    base_dir = os.path.dirname(tarFile)
    base_name = os.path.basename(tarFile)
    if '.' in base_name:
        base_name = base_name.split('.', 1)[0]
    out_path = '/'.join([base_dir, base_name])
    if not os.path.exists(out_path):
        df = tarfile.open(tarFile, mode='r')
        df.extractall(path=out_path)
        df.close()
    return out_path


def default_mapper(sample):
    '''
    map image bytes data to type needed by model input layer
    '''
    img, label = sample
    img = paddle.image.load_image_bytes(img)
    img = paddle.image.simple_transform(img, 256, 224, True)
    return img.flatten().astype('float32'), label


def reader_creator(data_file,
                   label_file,
                   setid_file,
                   flag,
                   mapper=default_mapper):
    '''
    1. extract 102flowers.tgz to 102flowers/
    2. merge images into batch files in 102flowers_batch/
    3. get a reader to read sample from batch file
    
    :param data_file: downloaded data file 
    :type data_file: string
    :param label_file: downloaded label file 
    :type label_file: string
    :param setid_file: downloaded setid file containing information
                        about how to split dataset
    :type setid_file: string
    :param flag: data set name (tstid|trnid|valid)
    :type flag: string
    :param mapper: a function to map image bytes data to type 
                    needed by model input layer
    :type mapper: callable
    :return: data reader
    :rtype: callable
    '''
    base_dir = os.path.dirname(data_file)
    tmp_dir = extract_file(data_file)
    file_list = create_batch(tmp_dir, label_file, setid_file, flag)

    def reader():
        for file in open(file_list):
            file = file.strip()
            batch = None
            with open(file, 'r') as f:
                batch = cPickle.load(f)
            data = batch['data']
            labels = batch['label']
            for sample, label in itertools.izip(data, batch['label']):
                yield sample, int(label)

    return paddle.reader.xmap(mapper, reader, cpu_count(), 1024 * 8)


def create_batch(data_dir,
                 label_file,
                 setid_file,
                 flag,
                 numPerBatch=1024,
                 nThread=16):
    batch_dir = data_dir + "_batch"
    labels = scio.loadmat(label_file)['labels'][0]
    indexes = scio.loadmat(setid_file)[flag][0]
    count = len(indexes)
    out_path = "%s/%s" % (batch_dir, flag)
    meta_file = "%s/%s.txt" % (batch_dir, flag)

    if os.path.exists(out_path):
        return meta_file
    else:
        os.makedirs(out_path)

    def batch(file_out, start, end):
        data = []
        labellist = []
        for index in indexes[start:end]:
            img_name = "%s/jpg/image_%05d.jpg" % (data_dir, index)
            with open(img_name, 'r') as f:
                data.append(f.read())
            labellist.append(labels[index - 1])
        output = {}
        output['label'] = labellist
        output['data'] = data
        cPickle.dump(
            output, open(file_out, 'w'), protocol=cPickle.HIGHEST_PROTOCOL)

    cur_id = 0
    file_id = 0
    while cur_id < count:
        thread = []
        for i in xrange(nThread):
            end_id = min(cur_id + numPerBatch, count)
            batch_file_name = "%s/batch_%05d" % (out_path, file_id)
            w = Process(target=batch, args=(batch_file_name, cur_id, end_id))
            w.daemon = True
            thread.append(w)
            cur_id = end_id
            file_id += 1
            if cur_id == count:
                break
        for t in thread:
            t.start()
        for t in thread:
            t.join()
    with open(meta_file, 'a') as meta:
        for file in os.listdir(out_path):
            meta.write(os.path.abspath("%s/%s" % (out_path, file)) + "\n")
    return meta_file


def train(mapper=default_mapper):
    '''
    Create flowers training set reader. 
    It returns a reader, each sample in the reader is   
    image pixels in [0, 1] and label in [1, 102] 
    translated from original color image by steps:
    1. resize to 256*256
    2. random crop to 224*224
    3. flatten
    :param mapper:  a function to map sample.
    :type mapper: callable
    :return: train data reader
    :rtype: callable
    '''
    return reader_creator(
        download(DATA_URL, 'flowers', DATA_MD5),
        download(LABEL_URL, 'flowers', LABEL_MD5),
        download(SETID_URL, 'flowers', SETID_MD5), 'trnid')


def test(mapper=default_mapper):
    '''
    Create flowers test set reader. 
    It returns a reader, each sample in the reader is   
    image pixels in [0, 1] and label in [1, 102] 
    translated from original color image by steps:
    1. resize to 256*256
    2. random crop to 224*224
    3. flatten
    :param mapper:  a function to map sample.
    :type mapper: callable
    :return: test data reader
    :rtype: callable
    '''
    return reader_creator(
        download(DATA_URL, 'flowers', DATA_MD5),
        download(LABEL_URL, 'flowers', LABEL_MD5),
        download(SETID_URL, 'flowers', SETID_MD5), 'tstid')


def valid():
    '''
    Create flowers validation set reader. 
    It returns a reader, each sample in the reader is   
    image pixels in [0, 1] and label in [1, 102] 
    translated from original color image by steps:
    1. resize to 256*256
    2. random crop to 224*224
    3. flatten
    '''
    return reader_creator(
        download(DATA_URL, 'flowers', DATA_MD5),
        download(LABEL_URL, 'flowers', LABEL_MD5),
        download(SETID_URL, 'flowers', SETID_MD5), 'valid')


def fetch():
    download(DATA_URL, 'flowers', DATA_MD5)
    download(LABEL_URL, 'flowers', LABEL_MD5)
    download(SETID_URL, 'flowers', SETID_MD5)


if __name__ == '__main__':
    for i in test()():
        pass
