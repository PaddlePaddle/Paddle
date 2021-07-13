# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import inspect
import sys
import re
import logging

logger = logging.getLogger()
if logger.handlers:
    # we assume the first handler is the one we want to configure
    console = logger.handlers[0]
else:
    console = logging.StreamHandler(sys.stderr)
    logger.addHandler(console)
console.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(funcName)s:%(lineno)d - %(levelname)s - %(message)s"))


def check_compatible(old_api_spec, new_api_spec):
    """
    check compatible, FullArgSpec
    """
    # 如果参数减少了，需要提醒关注
    if len(old_api_spec.args) > len(new_api_spec.args):
        return False
    # 参数改名了，也要提醒关注
    for idx in range(min(len(old_api_spec.args), len(new_api_spec.args))):
        if old_api_spec.args[idx] != new_api_spec.args[idx]:
            return False
    # 新增加了参数，必须提供默认值。以及不能减少默认值数量
    if (len(new_api_spec.args) - len(new_api_spec.defaults)) > (
            len(old_api_spec.args) - len(old_api_spec.defaults)):
        return False
    # 默认值必须相等
    for idx in range(
            min(len(old_api_spec.defaults), len(new_api_spec.defaults))):
        nidx = -1 - idx
        if (old_api_spec.defaults[nidx] != new_api_spec.defaults[nidx]):
            return False
    return True


def read_argspec_from_file(specfile):
    """
    read FullArgSpec from spec file
    """
    res_dict = {}
    patArgSpec = re.compile(
        r'^(paddle[^,]+)\s+\((ArgSpec.*),\s\(\'document\W*([0-9a-z]{32})')
    fullargspec_prefix = 'inspect.Full'
    for line in specfile.readlines():
        mo = patArgSpec.search(line)
        if mo and mo.group(2) != 'ArgSpec()':
            res_dict[mo.group(1)] = eval(fullargspec_prefix + mo.group(2))
    return res_dict


arguments = [
    # flags, dest, type, default, help
]


def parse_args():
    """
    Parse input arguments
    """
    global arguments
    parser = argparse.ArgumentParser(
        description='check api compatible across versions')
    parser.add_argument('--debug', dest='debug', action="store_true")
    parser.add_argument(
        'prev',
        type=argparse.FileType('r'),
        help='the previous version (the version from develop branch)')
    parser.add_argument(
        'post',
        type=argparse.FileType('r'),
        help='the post version (the version from PullRequest)')
    for item in arguments:
        parser.add_argument(
            item[0], dest=item[1], help=item[4], type=item[2], default=item[3])

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.prev and args.post:
        prev_spec = read_argspec_from_file(args.prev)
        post_spec = read_argspec_from_file(args.post)
        diff_api_names = []
        for as_post_name, as_post in post_spec.items():
            as_prev = prev_spec.get(as_post_name)
            if as_prev is None:  # the api is deleted
                continue
            if not check_compatible(as_prev, as_post):
                diff_api_names.append(as_post_name)
        if diff_api_names:
            print('\n'.join(diff_api_names))
