# -*- coding: UTF-8 -*-
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function
import os
import tarfile
import shutil
from collections import OrderedDict
import sys
import urllib
URLLIB=urllib
if sys.version_info >= (3, 0):
    import urllib.request
    URLLIB=urllib.request

__all__ = ["download", "ls"]

_pretrain = (('RoBERTa-zh-base', 'https://bert-models.bj.bcebos.com/chinese_roberta_wwm_ext_L-12_H-768_A-12.tar.gz'),
            ('RoBERTa-zh-large', 'https://bert-models.bj.bcebos.com/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16.tar.gz'),
            ('ERNIE-v2-en-base', 'https://ernie.bj.bcebos.com/ERNIE_Base_en_stable-2.0.0.tar.gz'),
            ('ERNIE-v2-en-large', 'https://ernie.bj.bcebos.com/ERNIE_Large_en_stable-2.0.0.tar.gz'),
            ('XLNet-cased-base', 'https://xlnet.bj.bcebos.com/xlnet_cased_L-12_H-768_A-12.tgz'),
            ('XLNet-cased-large', 'https://xlnet.bj.bcebos.com/xlnet_cased_L-24_H-1024_A-16.tgz'),
            ('ERNIE-v1-zh-base', 'https://baidu-nlp.bj.bcebos.com/ERNIE_stable-1.0.1.tar.gz'),
            ('ERNIE-v1-zh-base-max-len-512', 'https://ernie.bj.bcebos.com/ERNIE_1.0_max-len-512.tar.gz'),
            ('BERT-en-uncased-large-whole-word-masking', 'https://bert-models.bj.bcebos.com/wwm_uncased_L-24_H-1024_A-16.tar.gz'),
            ('BERT-en-cased-large-whole-word-masking', 'https://bert-models.bj.bcebos.com/wwm_cased_L-24_H-1024_A-16.tar.gz'),
            ('BERT-en-uncased-base', 'https://bert-models.bj.bcebos.com/uncased_L-12_H-768_A-12.tar.gz'),
            ('BERT-en-uncased-large', 'https://bert-models.bj.bcebos.com/uncased_L-24_H-1024_A-16.tar.gz'),
            ('BERT-en-cased-base', 'https://bert-models.bj.bcebos.com/cased_L-12_H-768_A-12.tar.gz'),
            ('BERT-en-cased-large','https://bert-models.bj.bcebos.com/cased_L-24_H-1024_A-16.tar.gz'),
            ('BERT-multilingual-uncased-base', 'https://bert-models.bj.bcebos.com/multilingual_L-12_H-768_A-12.tar.gz'),
            ('BERT-multilingual-cased-base', 'https://bert-models.bj.bcebos.com/multi_cased_L-12_H-768_A-12.tar.gz'),
            ('BERT-zh-base', 'https://bert-models.bj.bcebos.com/chinese_L-12_H-768_A-12.tar.gz'),)

_items = OrderedDict(_pretrain)

def _download(item, path, silent=False, convert=False):
    data_url = _items[item]
    if data_url == None:
        return
    if not silent:
        print('Downloading {} from {}...'.format(item, data_url))
    data_dir = path + '/' + item
    if not os.path.exists(data_dir):
        os.makedirs(os.path.join(data_dir))
    data_name = data_url.split('/')[-1]
    filename = data_dir + '/' + data_name

    # print process
    def _reporthook(count, chunk_size, total_size):
        bytes_so_far = count * chunk_size
        percent = float(bytes_so_far) / float(total_size)
        if percent > 1:
            percent = 1
        if not silent:
            print('\r>> Downloading... {:.1%}'.format(percent), end = "")
    
    URLLIB.urlretrieve(data_url, filename, reporthook=_reporthook)
    
    if not silent:
        print(' done!')
        print ('Extracting {}...'.format(data_name), end=" ")
    if os.path.exists(filename):
        tar = tarfile.open(filename, 'r')
        tar.extractall(path = data_dir)
        tar.close()
        os.remove(filename)
    if len(os.listdir(data_dir))==1:
        source_path = data_dir + '/' + data_name.split('.')[0]
        fileList = os.listdir(source_path)
        for file in fileList:
            filePath = os.path.join(source_path, file)
            shutil.move(filePath, data_dir)
        os.removedirs(source_path)
    if not silent:
        print ('done!')
    if convert:
        if not silent:
            print ('Converting params...', end=" ")
        _convert(data_dir, silent)


def _convert(path, silent=False):
    if os.path.isfile(path + '/params/__palminfo__'):
        if not silent:
            print ('already converted.')
    else:
        if os.path.exists(path + '/params/'):
            os.rename(path + '/params/', path + '/params1/')
            os.mkdir(path + '/params/')
            tar_model = tarfile.open(path + '/params/' + '__palmmodel__', 'w')
            tar_info = open(path + '/params/'+ '__palminfo__', 'w')
            for root, dirs, files in os.walk(path + '/params1/'):
                for file in files:
                    src_file = os.path.join(root, file)
                    tar_model.add(src_file, '__paddlepalm_' + file)
                    tar_info.write('__paddlepalm_' + file)
                    os.remove(src_file)
            tar_model.close()
            tar_info.close()
            os.removedirs(path + '/params1/') 
    if not silent:
        print ('done!')

def download(item='all', path='.'):
    """
    Args:
        item: the item to download.
        path: the target dir to download to. Default is `.`, means current dir.
    """
    # item = item.lower()
    # scope = scope.lower()
    if item != 'all':
        assert item in _items, '{} is not found. Support list: {}'.format(list(_items.keys()))
        _download(item, path)
    else:
        for item in _items.keys():
            _download(item, path)

def _ls():
    for item in _items.keys():
        print ('  => ' + item)

def ls(): 
    print ('Available pretrain models: ')
    _ls()
