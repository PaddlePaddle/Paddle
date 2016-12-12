#!/bin/env python2
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
Preprocess Movielens dataset, to get movie/user object.

Usage:
    ./preprocess.py <dataset_dir> <binary_filename> [--config=<config_file>]
    ./preprocess.py -h | --help

Options:
    -h --help               Show this screen.
    --version               Show version.
    --config=<config_file>  Get MetaData config file [default: config.json].
"""
import docopt
import os
import sys
import re
import collections

try:
    import cPickle as pickle
except ImportError:
    import pickle


class UniqueIDGenerator(object):
    def __init__(self):
        self.pool = collections.defaultdict(self.__next_id__)
        self.next_id = 0

    def __next_id__(self):
        tmp = self.next_id
        self.next_id += 1
        return tmp

    def __call__(self, k):
        return self.pool[k]

    def to_list(self):
        ret_val = [None] * len(self.pool)
        for k in self.pool.keys():
            ret_val[self.pool[k]] = k
        return ret_val


class SortedIDGenerator(object):
    def __init__(self):
        self.__key_set__ = set()
        self.dict = None

    def scan(self, key):
        self.__key_set__.add(key)

    def finish_scan(self, compare=None, key=None, reverse=False):
        self.__key_set__ = sorted(
            list(self.__key_set__), cmp=compare, key=key, reverse=reverse)
        self.dict = dict()
        for idx, each_key in enumerate(self.__key_set__):
            self.dict[each_key] = idx

    def __call__(self, key):
        return self.dict[key]

    def to_list(self):
        return self.__key_set__


class SplitFileReader(object):
    def __init__(self, work_dir, config):
        assert isinstance(config, dict)
        self.filename = config['name']
        self.delimiter = config.get('delimiter', ',')
        self.work_dir = work_dir

    def read(self):
        with open(os.path.join(self.work_dir, self.filename), 'r') as f:
            for line in f:
                line = line.strip()
                if isinstance(self.delimiter, unicode):
                    self.delimiter = str(self.delimiter)
                yield line.split(self.delimiter)

    @staticmethod
    def create(work_dir, config):
        assert isinstance(config, dict)
        if config['type'] == 'split':
            return SplitFileReader(work_dir, config)


class IFileReader(object):
    READERS = [SplitFileReader]

    def read(self):
        raise NotImplementedError()

    @staticmethod
    def create(work_dir, config):
        for reader_cls in IFileReader.READERS:
            val = reader_cls.create(work_dir, config)
            if val is not None:
                return val


class IDFieldParser(object):
    TYPE = 'id'

    def __init__(self, config):
        self.__max_id__ = -sys.maxint - 1
        self.__min_id__ = sys.maxint
        self.__id_count__ = 0

    def scan(self, line):
        idx = int(line)
        self.__max_id__ = max(self.__max_id__, idx)
        self.__min_id__ = min(self.__min_id__, idx)
        self.__id_count__ += 1

    def parse(self, line):
        return int(line)

    def meta_field(self):
        return {
            "is_key": True,
            'max': self.__max_id__,
            'min': self.__min_id__,
            'count': self.__id_count__,
            'type': 'id'
        }


class SplitEmbeddingDict(object):
    def __init__(self, delimiter):
        self.__id__ = UniqueIDGenerator()
        self.delimiter = delimiter

    def scan(self, multi):
        for val in multi.split(self.delimiter):
            self.__id__(val)

    def parse(self, multi):
        return map(self.__id__, multi.split(self.delimiter))

    def meta_field(self):
        return self.__id__.to_list()


class EmbeddingFieldParser(object):
    TYPE = 'embedding'

    NO_SEQUENCE = "no_sequence"
    SEQUENCE = "sequence"

    class CharBasedEmbeddingDict(object):
        def __init__(self, is_seq=True):
            self.__id__ = UniqueIDGenerator()
            self.is_seq = is_seq

        def scan(self, s):
            for ch in s:
                self.__id__(ch)

        def parse(self, s):
            return map(self.__id__, s) if self.is_seq else self.__id__(s[0])

        def meta_field(self):
            return self.__id__.to_list()

    class WholeContentDict(object):
        def __init__(self, need_sort=True):
            assert need_sort
            self.__id__ = SortedIDGenerator()
            self.__has_finished__ = False

        def scan(self, txt):
            self.__id__.scan(txt)

        def meta_field(self):
            if not self.__has_finished__:
                self.__id__.finish_scan()
                self.__has_finished__ = True
            return self.__id__.to_list()

        def parse(self, txt):
            return self.__id__(txt)

    def __init__(self, config):
        try:
            self.seq_type = config['type']['seq_type']
        except TypeError:
            self.seq_type = EmbeddingFieldParser.NO_SEQUENCE

        if config['dict']['type'] == 'char_based':
            self.dict = EmbeddingFieldParser.CharBasedEmbeddingDict(
                self.seq_type == EmbeddingFieldParser.SEQUENCE)
        elif config['dict']['type'] == 'split':
            self.dict = SplitEmbeddingDict(config['dict'].get('delimiter', ','))
        elif config['dict']['type'] == 'whole_content':
            self.dict = EmbeddingFieldParser.WholeContentDict(config['dict'][
                'sort'])
        else:
            print config
            assert False

        self.name = config['name']

    def scan(self, s):
        self.dict.scan(s)

    def meta_field(self):
        return {
            'name': self.name,
            'dict': self.dict.meta_field(),
            'type': 'embedding',
            'seq': self.seq_type
        }

    def parse(self, s):
        return self.dict.parse(s)


class OneHotDenseFieldParser(object):
    TYPE = 'one_hot_dense'

    def __init__(self, config):
        if config['dict']['type'] == 'split':
            self.dict = SplitEmbeddingDict(config['dict']['delimiter'])
        self.name = config['name']

    def scan(self, s):
        self.dict.scan(s)

    def meta_field(self):
        # print self.dict.meta_field()
        return {
            'dict': self.dict.meta_field(),
            'name': self.name,
            'type': 'one_hot_dense'
        }

    def parse(self, s):
        ids = self.dict.parse(s)
        retv = [0.0] * len(self.dict.meta_field())
        for idx in ids:
            retv[idx] = 1.0
        # print retv
        return retv


class FieldParserFactory(object):
    PARSERS = [IDFieldParser, EmbeddingFieldParser, OneHotDenseFieldParser]

    @staticmethod
    def create(config):
        if isinstance(config['type'], basestring):
            config_type = config['type']
        elif isinstance(config['type'], dict):
            config_type = config['type']['name']

        assert config_type is not None

        for each_parser_cls in FieldParserFactory.PARSERS:
            if config_type == each_parser_cls.TYPE:
                return each_parser_cls(config)
        print config


class CompositeFieldParser(object):
    def __init__(self, parser, extractor):
        self.extractor = extractor
        self.parser = parser

    def scan(self, *args, **kwargs):
        self.parser.scan(self.extractor.extract(*args, **kwargs))

    def parse(self, *args, **kwargs):
        return self.parser.parse(self.extractor.extract(*args, **kwargs))

    def meta_field(self):
        return self.parser.meta_field()


class PositionContentExtractor(object):
    def __init__(self, pos):
        self.pos = pos

    def extract(self, line):
        assert isinstance(line, list)
        return line[self.pos]


class RegexPositionContentExtractor(PositionContentExtractor):
    def __init__(self, pos, pattern, group_id, strip=True):
        PositionContentExtractor.__init__(self, pos)
        pattern = pattern.strip()
        self.pattern = re.compile(pattern)
        self.group_id = group_id
        self.strip = strip

    def extract(self, line):
        line = PositionContentExtractor.extract(self, line)
        match = self.pattern.match(line)
        # print line, self.pattern.pattern, match
        assert match is not None
        txt = match.group(self.group_id)
        if self.strip:
            txt.strip()
        return txt


class ContentExtractorFactory(object):
    def extract(self, line):
        pass

    @staticmethod
    def create(config):
        if 'pos' in config:
            if 'regex' not in config:
                return PositionContentExtractor(config['pos'])
            else:
                extra_args = config['regex']
                return RegexPositionContentExtractor(
                    pos=config['pos'], **extra_args)


class MetaFile(object):
    def __init__(self, work_dir):
        self.work_dir = work_dir
        self.obj = dict()

    def parse(self, config):
        config = config['meta']

        ret_obj = dict()
        for key in config.keys():
            val = config[key]
            assert 'file' in val
            reader = IFileReader.create(self.work_dir, val['file'])
            assert reader is not None
            assert 'fields' in val and isinstance(val['fields'], list)
            fields_config = val['fields']
            field_parsers = map(MetaFile.__field_config_mapper__, fields_config)

            for each_parser in field_parsers:
                assert each_parser is not None

            for each_block in reader.read():
                for each_parser in field_parsers:
                    each_parser.scan(each_block)

            metas = map(lambda x: x.meta_field(), field_parsers)
            # print metas
            key_index = filter(
                lambda x: x is not None,
                map(lambda (idx, meta): idx if 'is_key' in meta and meta['is_key'] else None,
                    enumerate(metas)))[0]

            key_map = []
            for i in range(min(key_index, len(metas))):
                key_map.append(i)
            for i in range(key_index + 1, len(metas)):
                key_map.append(i)

            obj = {'__meta__': {'raw_meta': metas, 'feature_map': key_map}}

            for each_block in reader.read():
                idx = field_parsers[key_index].parse(each_block)
                val = []
                for i, each_parser in enumerate(field_parsers):
                    if i != key_index:
                        val.append(each_parser.parse(each_block))
                obj[idx] = val
            ret_obj[key] = obj
        self.obj = ret_obj
        return ret_obj

    @staticmethod
    def __field_config_mapper__(conf):
        assert isinstance(conf, dict)
        extrator = ContentExtractorFactory.create(conf)
        field_parser = FieldParserFactory.create(conf)
        assert extrator is not None
        assert field_parser is not None
        return CompositeFieldParser(field_parser, extrator)

    def dump(self, fp):
        pickle.dump(self.obj, fp, pickle.HIGHEST_PROTOCOL)


def preprocess(binary_filename, dataset_dir, config, **kwargs):
    assert isinstance(config, str)
    with open(config, 'r') as config_file:
        file_loader = None
        if config.lower().endswith('.yaml'):
            import yaml
            file_loader = yaml
        elif config.lower().endswith('.json'):
            import json
            file_loader = json
        config = file_loader.load(config_file)
    meta = MetaFile(dataset_dir)
    meta.parse(config)
    with open(binary_filename, 'wb') as outf:
        meta.dump(outf)


if __name__ == '__main__':
    args = docopt.docopt(__doc__, version='0.1.0')
    kwargs = dict()
    for key in args.keys():
        if key != '--help':
            param_name = key
            assert isinstance(param_name, str)
            param_name = param_name.replace('<', '')
            param_name = param_name.replace('>', '')
            param_name = param_name.replace('--', '')
            kwargs[param_name] = args[key]
    preprocess(**kwargs)
