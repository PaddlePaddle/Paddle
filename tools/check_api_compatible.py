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

#1) 原api有的参数新api都有，且顺序一致
#2）无默认参数api数量，原api大于等于新api


# apilist = [paddle.cast, paddle.nn.functional.relu]
def get_api_dict(api):
    api_dict = {}
    api_argcount = api.__code__.co_argcount  #输入参数数量
    api_dict['count'] = api_argcount
    api_argnames = api.__code__.co_varnames  #输入参数名称tuple
    api_dict['args'] = api_argnames
    api_defaults = api.__defaults__  #输入参数默认值tuple
    api_dict['args_defaults'] = api_defaults
    return api_dict


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
    for line in specfile.readlines():
        mo = patArgSpec.search(line)
        if mo:
            res_dict[mo.group(1)] = mo.group(2)
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
        type=str,
        help='the previous version (the version from develop branch)',
        default=None)
    parser.add_argument(
        'post',
        type=str,
        help='the post version (the version from PullRequest)',
        default=None)
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
