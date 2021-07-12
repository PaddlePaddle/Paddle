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
import paddle

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


def check_compatible(old_api_dict, new_api_dict):
    old_argcount = old_api_dict['count']
    old_argnames = old_api_dict['args']
    old_argdefaults = old_api_dict['args_defaults']
    old_dn = 0 if (old_argdefaults == None) else len(old_argdefaults)

    new_argcount = new_api_dict['count']
    new_argnames = new_api_dict['args']
    new_argdefaults = new_api_dict['args_defaults']
    new_dn = 0 if (new_argdefaults == None) else len(new_argdefaults)

    if old_argcount > new_argcount:
        return False
    for idx in range(min(len(old_argnames), len(new_argnames))):
        if old_argnames[idx] != new_argnames[idx]:
            return False
    if ((new_argcount - new_dn) > (old_argcount - old_dn)):
        return False
    for idx in range(
            max((new_argcount - new_dn), (old_argcount - old_dn)),
            min(new_argcount, old_argcount)):
        newargidx = idx - (new_argcount - new_dn)
        oldargidx = idx - (old_argcount - old_dn)
        if (new_argdefaults[newargidx] != old_argdefaults[oldargidx]):
            return False
    return True


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
