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
import six
import re
import json
import argparse
import itertools

from paddle.fluid.incubate.atarashi.types import RunConfig
from paddle.fluid.incubate.atarashi import log


def ArgumentParser(name):
    parser = argparse.ArgumentParser('DAN model on paddle')
    #parser.add_argument('--param_dir', type=str)
    parser.add_argument('--run_config', type=str, default='')
    parser.add_argument('--hparam', type=str, default='')
    #parser.add_argument('--train_dir', type=str)
    parser.add_argument('--batch_size', type=int)
    return parser


def _get_dict_from_environ_or_json_or_file(args, env_name):
    if args is None:
        s = os.environ.get(env_name)
    else:
        s = args
        if os.path.exists(s):
            s = open(s).read()
    if isinstance(s, six.string_types):
        try:
            r = eval(s)
        except SyntaxError as e:
            raise ValueError('json parse error: %s \n>Got json: %s' %
                             (repr(e), s))
        return r
    else:
        return s  #None


def parse_runconfig(args=None):
    d = _get_dict_from_environ_or_json_or_file(args.run_config,
                                               'ATARASHI_RUNCONFIG')
    if d is None:
        raise ValueError('run_config not found')
    return RunConfig(**d)


def parse_hparam(args=None):
    d = _get_dict_from_environ_or_json_or_file(args.hparam,
                                               'ATARASHI_RUNCONFIG')
    if d is None:
        raise ValueError('hparam not found')
    return d


def flatten(s):
    assert is_struture(s)
    schema = [len(ss) for ss in s]
    flt = list(itertools.chain(*s))
    return flt, schema


def unflatten(structure, schema):
    start = 0
    res = []
    for _range in schema:
        res.append(structure[start:start + _range])
        start += _range
    return res


def is_struture(s):
    return isinstance(s, list) or isinstance(s, tuple)


def map_structure(func, s):
    if isinstance(s, list) or isinstance(s, tuple):
        return [map_structure(func, ss) for ss in s]
    elif isinstance(s, dict):
        return {k: map_structure(func, v) for k, v in six.iteritems(s)}
    else:
        return func(s)
