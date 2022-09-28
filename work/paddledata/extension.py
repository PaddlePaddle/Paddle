import os
import tarfile
import zipfile
import py7zr
import rarfile
import wget
import paramiko
import pandas as pd
import numpy as np
from paddle.io import Dataset
from paddle.io import IterableDataset
from paddle.io import DataLoader

from paddledata.file_extract import *
from paddledata.text_decoding import *
from paddledata.remote_operation import *


class DataPipes(object):
    def __init__(self, pipelines):
        self.pipelines = pipelines

    def __call__(self, data):
        for pipe in self.pipelines:
            try:
                data = pipe(data)
            except Exception as e:
                print("fail to perform processing [{}] with error: {}".format(pipe, e))
                raise e
        return data


class MyDataLoader(DataLoader):
    def __init__(self, dataset, datapipes):
        self.dataset = dataset
        self.datapipes = datapipes
        self.loader = self.datapipes(self.dataset)
    
    def __iter__(self):
        for (data, label) in self.loader:
            yield data, label

