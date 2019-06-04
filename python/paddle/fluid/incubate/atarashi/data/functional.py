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
from __future__ import absolute_import
from __future__ import unicode_literals

import sys
import logging
import os
import itertools
import random
import numpy as np

import gzip
import struct
import functools
import six
from six.moves import zip, map

import paddle.fluid as F
import paddle.fluid.layers as L
from paddle.fluid.incubate.atarashi.util import map_structure
from paddle.fluid.incubate.atarashi import log

__all__ = ['Dataset', 'interleave_func']


def open_gz(filename):
    def gen():
        with gzip.open(filename, 'rb') as f:
            while True:
                data = f.read(struct.calcsize('i'))
                if not len(data):
                    raise StopIteration
                l, = struct.unpack('i', data)
                data = f.read(l)
                yield data

    return gen


def open_file(filename):
    def gen():
        with open(filename, 'rb') as f:
            for line in f:
                yield line

    return gen


def shuffle_func(dataset, buffer_size):
    buf = []

    def gen():
        iterable = iter(dataset)
        while True:
            buf = list(itertools.islice(iterable,
                                        buffer_size))  #cache this iterator
            if len(buf):
                np.random.shuffle(buf)
                for item in buf:
                    yield item
            else:
                raise StopIteration

    return Dataset.from_generator(gen)


def interleave_func(iterable, map_fn, cycle_length, block_length):
    def gen():
        ls = itertools.tee(iterable, cycle_length)
        buf = []
        for i, j in enumerate(ls):
            j = itertools.islice(j, i, None, cycle_length)
            j = map(map_fn, j)
            j = (jjj for jj in j for jjj in jj)  #flatten
            buf.append(j)

        for tup in six.moves.zip_longest(*buf):
            for ii in (i for i in tup if i is not None):
                yield ii

    return Dataset.from_generator(gen)


def repeat_func(dataset, n):
    def gen():
        cnt = 0
        iterable = iter(dataset)
        while True:
            for item in iterable:
                yield item
            cnt += 1
            if cnt == n:
                break
            else:
                iterable = iter(dataset)

    return Dataset.from_generator(gen)


def map_map_func(dataset, map_fn):
    def gen():
        for i in dataset:
            if isinstance(i, tuple) or isinstance(i, list):
                yield map_fn(*i)
            else:
                yield map_fn(i)

    return Dataset.from_generator(gen)


def padded_batch_func(dataset, batch_size, pad_value):
    def gen():
        iterable = iter(dataset)
        while True:
            buf = list(itertools.islice(iterable, batch_size))
            if not len(buf):
                raise StopIteration
            buf = list(zip(*buf))  # transpose
            padded = []
            assert len(buf) == len(
                pad_value), 'pad_value [%d] != element size[%d]' % (
                    len(buf), len(pad_value))
            for e, pv in zip(buf, pad_value):
                elem = e[0]
                if (not np.isscalar(elem)) and elem.shape != ():
                    max_len = max(map(len, e))
                    e = map(lambda i: np.pad(i, [0, max_len - len(i)], 'constant', constant_values=pv), e)
                padded.append(np.stack(e))
            yield padded

    return Dataset.from_generator(gen)


class Dataset(object):
    @classmethod
    def from_generator(cls, gen, data_shapes=None, data_types=None):
        ret = Dataset()
        ret.generator = gen
        ret.data_shapes = data_shapes
        ret.data_types = data_types
        return ret

    @classmethod
    def from_file(cls, filename):
        gen = open_file(filename)
        ret = Dataset()
        ret.generator = gen
        ret.data_shapes = []
        ret.data_types = str
        return ret

    @classmethod
    def from_gz_file(cls, filename):
        gen = open_gz(filename)
        ret = Dataset()
        ret.generator = gen
        ret.data_shapes = []
        ret.data_types = str
        return ret

    @classmethod
    def from_iterable(cls, iterable):
        def gen():
            for i in iterable:
                yield i

        ret = Dataset()
        ret.generator = gen
        ret.data_shapes = []
        ret.data_types = str
        return ret

    def __init__(self):
        self.name = None
        self.data_shapes = None
        self.data_types = None
        self.generator = None
        self.pyreader = None

    def __iter__(self):
        def gen():
            try:
                for i in self.generator():
                    yield i
            except Exception as e:
                log.exception(e)

        return gen()

    #def __call__(self):
    #    return self.generator()

    def placeholders(self):
        if self.data_shapes is None or self.data_types is None:
            raise ValueError(
                'try making placeholder from a shape/types undefined dataset')
        ret = []
        for i, (shape,
                types) in enumerate(zip(self.data_shapes, self.data_types)):
            ret.append(
                L.data(
                    str(i), shape=shape, append_batch_size=False, dtype=types))
        return ret

    def features(self):
        '''start point of net building. call this in a program scope'''
        assert self.name is not None, 'unnamed Dataset'
        if len(self.data_shapes) != len(self.data_types):
            raise ValueError(
                'Dataset shapes and types not match: shape:%s types%s' %
                (repr(self.data_shapes), repr(self.data_types)))
        self.pyreader = L.py_reader(
            50,
            shapes=self.data_shapes,
            dtypes=self.data_types,
            lod_levels=[0] * len(self.data_shapes),
            name=self.name,
            use_double_buffer=True, )
        features = L.read_file(self.pyreader)
        return features

    def start(self):
        assert self.pyreader is not None, 'use Dataset.features to build net first, then start dataset'
        return PyreaderContext(self.pyreader, self.generator)

    def apply(self, transform_func):
        #input_shapes = transform_func.input_shapes
        #input_types = transform_func.input_types
        #data_shapes = transform_func.data_shapes
        #data_types = transform_func.data_types
        #assert input_shapes == self.data_shapes
        #assert input_types = self.data_types
        ret = transform_func(self)
        if self.name is not None:
            ret.name = self.name
        #ret.data_shapes = data_shapes
        #ret.data_types = data_types
        return ret

    def shuffle(self, buffer_size):
        func = functools.partial(shuffle_func, buffer_size=buffer_size)
        return self.apply(func)

    def repeat(self, n=-1):
        func = functools.partial(repeat_func, n=n)
        return self.apply(func)

    def map(self, map_fn):
        func = functools.partial(map_map_func, map_fn=map_fn)
        return self.apply(func)

    def padded_batch(self, batch_size, pad_value):
        func = functools.partial(
            padded_batch_func, batch_size=batch_size, pad_value=pad_value)
        return self.apply(func)


class PyreaderContext(object):
    def __init__(self, pyreader, dataset):
        pyreader.decorate_tensor_provider(dataset)
        self._pyreader = pyreader

    def __enter__(self):
        self._pyreader.start()
        return self

    def __exit__(self, err_type, err_value, trace):
        self._pyreader.reset()
