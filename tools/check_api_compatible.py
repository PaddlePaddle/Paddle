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

import paddle

#1) 原api有的参数新api都有，且顺序一致
#2）无默认参数api数量，原api大于等于新api

# apilist = [paddle.cast, paddle.nn.functional.relu]

all_apis_dict = {}
for api in apilist:
    api_dict = {}
    api_argcount = api.__code__.co_argcount  #输入参数数量
    api_dict['count'] = api_argcount
    api_argnames = api.__code__.co_varnames  #输入参数名称tuple
    api_dict['args'] = api_argnames
    api_defaults = api.__defaults__  #输入参数默认值tuple
    api_dict['args_defaults'] = api_defaults
    all_apis_dict[api.__name__] = api_dict


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


old_api_dict = {'count': 3, 'args': ('x', 'y', 'name'), 'args_defaults': None}

new_api_dict = {
    'count': 4,
    'args': ('y', 'x', 'name'),
    'args_defaults': (None, )
}
print(check_compatible(old_api_dict, new_api_dict))
