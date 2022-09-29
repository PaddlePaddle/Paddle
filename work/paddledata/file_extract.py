import os
import tarfile
import zipfile
import py7zr
import rarfile

import numpy as np
from paddle.io import Dataset
from paddle.io import IterableDataset


def untar(file_dir, save_dir):
    """
    支持tar,tar.gz,tgz文件解压
    """
    files = tarfile.open(file_dir)
    for file in files.getnames():
        files.extract(file, save_dir)


def unzip(file_dir, save_dir):
    """
    支持zip文件解压
    """
    is_zip = zipfile.is_zipfile(file_dir)
    if is_zip:   
        files = zipfile.ZipFile(file_dir, 'r')
        for file in files.namelist():
            files.extract(file, save_dir)     
    else:
        print('This is not zip')


def un7z(file_dir, save_dir):
    """
    支持7z文件解压
    """
    files = py7zr.SevenZipFile(file_dir, mode='r')
    files.extractall(path=save_dir)
    files.close()


def unrar(file_dir, save_dir):
    """
    暂不支持rar文件解压
    """
    files = rarfile.RarFile(file_dir, mode='r')
    files.extractall(path=save_dir)
    files.close()


"""
untar(file_dir='demo3-file-decompression/test.tar', save_dir='./data/decompression/tar')
untar(file_dir='demo3-file-decompression/test.tar.gz', save_dir='./data/decompression/targz')
untar(file_dir='demo3-file-decompression/test.tgz', save_dir='./data/decompression/tgz')
unzip(file_dir='demo3-file-decompression/test.zip', save_dir='./data/decompression/zip')
un7z(file_dir='demo3-file-decompression/test.7z', save_dir='./data/decompression/7z')
# unrar(file_dir='demo3-file-decompression/test.rar', save_dir='./data/decompression/rar')
"""


def _get_file_format(file_dir):
    if file_dir[-4:] == '.tar':
        file_format = 'tar'
    elif file_dir[-7:] == '.tar.gz':
        file_format = 'tar.gz'
    elif file_dir[-4:] == '.tgz':
        file_format = 'tgz'
    elif file_dir[-4:] == '.zip':
        file_format = 'zip'
    elif file_dir[-3:] == '.7z':
        file_format = '7z'
    elif file_dir[-4:] == '.rar':
        file_format = 'rar'
    else:
        raise ValueError('传入的文件不支持解压缩')
    return file_format


class Extractor(object):
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.cur_dir = os.path.abspath(self.save_dir)
    
    def __iter__(self):
        for f in self.files:
            yield f
    
    def __call__(self, dataset_source):
        self.dataset_source = dataset_source
        for file_dir in dataset_source:
            try:
                file_format = _get_file_format(file_dir)
                self.file_format = file_format.lower()
                if self.file_format in ['tar', 'tar.gz', 'tgz']:
                    self.extract = untar
                elif self.file_format in ['zip']:
                    self.extract = unzip
                elif self.file_format in ['7z']:
                    self.extract = un7z
                else:
                    raise ValueError('传入的参数\'file_format\'不支持解压缩')
                self.extract(file_dir=file_dir, save_dir=self.save_dir)
            except:
                pass
        files = os.listdir(self.save_dir)
        self.files = [os.path.join(self.cur_dir, f) for f in files]
        return self

