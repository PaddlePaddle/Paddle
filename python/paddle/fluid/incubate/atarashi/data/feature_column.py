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

import os
import sys
import struct
from six.moves import zip, map
import itertools
import gzip
from functools import partial
import multiprocessing

import numpy as np
from glob import glob
from paddle.fluid.incubate.atarashi import log
from paddle.fluid.incubate.atarashi.data import Dataset, example_pb2, interleave_func

__all__ = [
    'FeatureColumns', 'TextColumn', 'TextIDColumn', 'LabelColumn',
    'basic_tokenizer'
]


def basic_tokenizer(sen):
    seg = sen.split(b' ')
    seg = filter(lambda i: i != b' ', seg)
    return seg


class Column(object):
    def __init__(self, name):
        pass

    def raw_to_proto(self, raw):
        return example_pb2.Feature()

    @property
    def output_shapes(self):
        pass

    @property
    def output_types(self):
        raise NotImplementedError()

    def proto_to_instance(self, proto):
        raise NotImplementedError()

    def raw_to_instance(self, raw):
        raise NotImplementedError()


class LabelColumn(Column):
    def __init__(self, name='text'):
        self.name = name

    @property
    def output_shapes(self):
        return [1]

    @property
    def output_types(self):
        return 'int64'

    def raw_to_proto(self, raw):
        ids = [int(raw)]
        fe = example_pb2.Feature(
            tag=self.name, int64_list=example_pb2.Int64List(value=ids))
        return fe

    def proto_to_instance(self, feature):
        ret = np.array(feature.int64_list.value[0], dtype=np.int64)
        return ret

    def raw_to_instance(self, raw):
        return int(raw)


class TextColumn(Column):
    def __init__(self,
                 name='text',
                 vocab_file=None,
                 vocab_list=None,
                 unk_id=1,
                 tokenizer=basic_tokenizer):
        self.name = name
        self.tokenizer = tokenizer
        self.unk_id = unk_id
        assert vocab_file or vocab_list
        if vocab_file:
            self.vocab = {
                j.strip(): i
                for i, j in enumerate(open(vocab_file, 'rb').readlines())
            }
        if vocab_list:
            self.vocab = vocab_list

    @property
    def output_shapes(self):
        return [-1]

    @property
    def output_types(self):
        return 'int64'

    def raw_to_proto(self, raw):
        ids = [self.vocab.get(s, self.unk_id) for s in self.tokenizer(raw)]
        fe = example_pb2.Feature(
            tag=self.name, int64_list=example_pb2.Int64List(value=ids))
        return fe

    def proto_to_instance(self, feature):
        ret = np.array(feature.int64_list.value, dtype=np.int64)
        return ret


class TextIDColumn(Column):
    def __init__(self, name='text'):
        self.name = name

    @property
    def output_shapes(self):
        return [-1]

    @property
    def output_types(self):
        return 'int64'

    def raw_to_proto(self, raw):
        ids = [int(s) for s in raw.split(b' ')]
        fe = example_pb2.Feature(
            tag=self.name, int64_list=example_pb2.Int64List(value=ids))
        return fe

    def proto_to_instance(self, feature):
        ret = np.array(feature.int64_list.value, dtype=np.int64)
        return ret

    def raw_to_instance(self, raw):
        ret = np.array([int(i) for i in raw.split(b' ')], dtype=np.int64)
        return ret


class FeatureColumns(object):
    def __init__(self, columns, pad_id=0):
        self._pool = None
        self._columns = columns

    def __del__(self):
        if self._pool is not None:
            self._pool.terminate()

    def pool(self):
        if self._pool is None:
            self._pool = multiprocessing.Pool()
        return self._pool

    def raw_files(self, raw_dir):
        return [os.path.join(raw_dir, p) for p in os.listdir(raw_dir)]

    def gz_files(self, gz_dir):
        return None if gz_dir is None else [
            os.path.join(gz_dir, p) for p in os.listdir(gz_dir)
        ]

    def _make_gz_dataset(self, raw_dir, gz_dir):
        assert raw_dir or gz_dir, 'data_dir not specified when using gz mode'
        if raw_dir is not None:
            assert os.path.exists(raw_dir)
            raw_file = os.listdir(raw_dir)
        if gz_dir is None:
            gz_dir = '%s_gz' % raw_dir.strip('/')

        if not os.path.exists(gz_dir):
            os.mkdir(gz_dir)

        if raw_dir is not None:
            if len(raw_file) != 0:
                log.debug('try making gz')
                pool = self.pool()
                args = [(os.path.join(raw_dir, f), os.path.join(gz_dir, f),
                         self._columns, b'\t') for f in raw_file]
                pool.map(_make_gz, args)
            else:
                assert len(
                    os.listdir(gz_dir)
                ) != 0, 'cant find gz file or raw-txt file at [%s] and [%s]' % (
                    raw_dir, gz_dir)
        return gz_dir

    def _read_gz_dataset(self, gz_files, shuffle=False, repeat=True, **kwargs):
        if len(gz_files) == 0:
            raise ValueError('reading gz from empty file list: %s' % gz_files)
        log.info('reading gz from %s' % '\n'.join(gz_files))
        dataset = Dataset.from_iterable(gz_files)
        if repeat:
            dataset = dataset.repeat()
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(gz_files))
        fn = partial(
            interleave_func,
            map_fn=lambda filename: Dataset.from_gz_file(filename),
            cycle_length=len(gz_files),
            block_length=1)
        dataset = dataset.apply(fn)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)

        def _parse_gz(record_str):  # function that takes python_str as input
            ex = example_pb2.Example()
            ex.ParseFromString(record_str)
            ret = [
                column.proto_to_instance(feature)
                for feature, column in zip(ex.features, self._columns)
            ]
            return ret

        dataset = dataset.map(_parse_gz)
        return dataset

    def _read_txt_dataset(self,
                          data_files,
                          shuffle=False,
                          repeat=True,
                          **kwargs):
        log.info('reading raw files from %s' % '\n'.join(data_files))
        dataset = Dataset.from_iterable(data_files)
        if repeat:
            dataset = dataset.repeat()
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(data_files))

        fn = partial(
            interleave_func,
            map_fn=lambda filename: Dataset.from_file(filename),
            cycle_length=len(data_files),
            block_length=1)
        dataset = dataset.apply(fn)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)

        def _parse_txt_file(
                record_str):  # function that takes python_str as input
            features = record_str.strip(b'\n').split(b'\t')
            ret = [
                column.raw_to_instance(feature)
                for feature, column in zip(features, self._columns)
            ]
            return ret

        dataset = dataset.map(_parse_txt_file)
        return dataset

    def _prepare_dataset(self,
                         dataset,
                         map_func_before_batch=None,
                         map_func_after_batch=None,
                         shuffle_buffer_size=None,
                         batch_size=1,
                         pad_id=0,
                         prefetch=None,
                         **kwargs):

        if map_func_before_batch is not None:
            dataset = dataset.map(map_func_before_batch)
        if batch_size:
            dataset = dataset.padded_batch(batch_size, pad_id)
        if map_func_after_batch is not None:
            dataset = dataset.map(map_func_after_batch)
        return dataset

    def build_dataset(self,
                      name,
                      use_gz=True,
                      data_dir=None,
                      gz_dir=None,
                      data_file=None,
                      **kwargs):
        if use_gz:
            gz_dir = self._make_gz_dataset(data_dir, gz_dir)
            gz_files = self.gz_files(gz_dir)
            ds = self._read_gz_dataset(gz_files, **kwargs)
        else:
            if data_dir is not None:
                data_files = self.raw_files(data_dir)
            elif data_file is not None:
                data_files = [data_file]
            else:
                raise ValueError('data_dir or data_files not specified')
            ds = self._read_txt_dataset(data_files, **kwargs)
        ds.name = name
        return ds


def _make_gz(args):
    try:
        from_file, to_file, columns, sep = args
        if os.path.exists(to_file):
            return
        with open(from_file, 'rb') as fin, gzip.open(to_file, 'wb') as fout:
            log.debug('making gz %s => %s' % (from_file, to_file))
            for i, line in enumerate(fin):
                line = line.strip(b'\n').split(sep)
                #if i % 10000 == 0:
                #    log.debug('making gz %s => %s [%d]' % (from_file, to_file, i))
                if len(line) != len(columns):
                    log.error('columns not match at %s, got %d, expect %d' %
                              (from_file, len(line), len(columns)))
                    continue
                features = []
                for l, c in zip(line, columns):
                    features.append(c.raw_to_proto(l))
                example = example_pb2.Example(features=features)
                serialized = example.SerializeToString()
                l = len(serialized)
                data = struct.pack('i%ds' % l, l, serialized)
                fout.write(data)
            log.debug('done making gz %s => %s' % (from_file, to_file))
    except Exception as e:
        log.exception(e)
        raise e
