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
config_generator.py

Usage:
    ./config_generator.py <config_file> [--output_format=<output_format>]
    ./config_generator.py -h | --help

Options:
    -h --help                           Show this screen.
    --output_format=<output_format>     Output Config format(json or yaml) [default: json].
"""

import json
import docopt
import copy

DEFAULT_FILE = {"type": "split", "delimiter": ","}

DEFAULT_FIELD = {
    "id": {
        "type": "id"
    },
    "gender": {
        "name": "gender",
        "type": "embedding",
        "dict": {
            "type": "char_based"
        }
    },
    "age": {
        "name": "age",
        "type": "embedding",
        "dict": {
            "type": "whole_content",
            "sort": True
        }
    },
    "occupation": {
        "name": "occupation",
        "type": "embedding",
        "dict": {
            "type": "whole_content",
            "sort": "true"
        }
    },
    "title": {
        "regex": {
            "pattern": r"^(.*)\((\d+)\)$",
            "group_id": 1,
            "strip": True
        },
        "name": "title",
        "type": {
            "name": "embedding",
            "seq_type": "sequence",
        },
        "dict": {
            "type": "char_based"
        }
    },
    "genres": {
        "type": "one_hot_dense",
        "dict": {
            "type": "split",
            "delimiter": "|"
        },
        "name": "genres"
    }
}


def merge_dict(master_dict, slave_dict):
    return dict(((k, master_dict.get(k) or slave_dict.get(k))
                 for k in set(slave_dict) | set(master_dict)))


def main(filename, fmt):
    with open(filename, 'r') as f:
        conf = json.load(f)
        obj = dict()
        for k in conf:
            val = conf[k]
            file_dict = val['file']
            file_dict = merge_dict(file_dict, DEFAULT_FILE)

            fields = []
            for pos, field_key in enumerate(val['fields']):
                assert isinstance(field_key, basestring)
                field = copy.deepcopy(DEFAULT_FIELD[field_key])
                field['pos'] = pos
                fields.append(field)
            obj[k] = {"file": file_dict, "fields": fields}
    meta = {"meta": obj}
    # print meta
    if fmt == 'json':

        def formatter(x):
            import json
            return json.dumps(x, indent=2)
    elif fmt == 'yaml':

        def formatter(x):
            import yaml
            return yaml.safe_dump(x, default_flow_style=False)
    else:
        raise NotImplementedError("Dump format %s is not implemented" % fmt)

    print formatter(meta)


if __name__ == '__main__':
    args = docopt.docopt(__doc__, version="0.1.0")
    main(args["<config_file>"], args["--output_format"])
