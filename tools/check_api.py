#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
import json
import argparse
import traceback
import inspect
import paddle.fluid as fluid


def parse_args():
    parser = argparse.ArgumentParser("Python API Check Tool.")
    parser.add_argument(
        '--op',
        type=str,
        default='check',
        choices=['save', 'check'],
        help='Operations to choose')
    parser.add_argument(
        '--dir',
        type=str,
        default='api_args',
        help='Dir to save or load api infos.')
    args = parser.parse_args()
    return args


def get_api_args(module):
    '''Return Dict. keys: function/class.method
                     values: dict of args info'''
    # get global function's args 
    args_dict = {}
    funs = inspect.getmembers(module, inspect.isfunction)
    for name, value in funs:
        #tuple (args, varargs, keywords, defaults) is returned
        tup = inspect.getargspec(value)
        args_dict[name] = {
            "args": tup[0],  #args is a list of the parameter names.
            "varargs": tup[1],  #varargs are the names of the * parameters
            "keywords": tup[2],  #keywords are the names of the ** parameters
            "defaults": str(deal_dynamic_value(tup[3]))  #default values
        }
    # get class method's args
    cls = inspect.getmembers(module, inspect.isclass)
    for cname, cvalue in cls:
        if hasattr(module, '__all__') and cname not in module.__all__:
            #exclude classes imported in
            continue
        for fname, fvalue in inspect.getmembers(cvalue, inspect.ismethod):
            if fname.startswith('_') and fname != '__init__':
                #ignore private methods of class, not including init method
                continue
            try:
                tup = inspect.getargspec(fvalue)
            except TypeError as e:
                print fvalue, e
                continue
            args_dict['%s.%s' % (cname, fname)] = {
                "args": tup[0],
                "varargs": tup[1],
                "keywords": tup[2],
                "defaults": str(deal_dynamic_value(tup[3]))
            }
    return args_dict


def deal_dynamic_value(tup):
    ''' repalce dynamic function obj with function name,
       which is like <function round_robin at 0x7f33b1eb4aa0>
       each time the function addr has changed. '''
    values = []
    if not tup:
        return
    for i in tup:
        if inspect.isfunction(i):
            values.append(i.__name__)
        else:
            values.append(i)
    return tuple(values)


def get_all_files(base_dir):
    '''recursively traversing fluid directories to 
         get all module files in fluid'''
    file_list = []
    for fpathe, dirs, fs in os.walk(base_dir):
        for f in fs:
            # exclude hidden file
            if f.startswith('.'):
                continue
            file_list.append(os.path.join(fpathe, f))
    return file_list


def save_api(module_str, dest_dir):
    '''the output file's name is like 'fluid.layers.nn'
       the content is a dict
           key: function/class.method name
           value: it's args info'''
    try:
        args_dict = get_api_args(eval(module_str))
    except AttributeError as e:
        print e
        return
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    with open('%s/%s' % (dest_dir, module_str), 'w') as f:
        f.write(json.dumps(args_dict))


def check_api(module_str, dest_dir):
    '''get current api's info, 
        then compare with info loaded from saved'''
    try:
        args_dict = get_api_args(eval(module_str))
    except AttributeError as e:
        print e
        return
    with open('%s/%s' % (dest_dir, module_str), 'r') as f:
        latest_args_dict = json.load(f)
    if args_dict == latest_args_dict:
        pass
    else:
        for k, v in args_dict.items():
            # if a api added, KeyError occurs, should it be catched? 
            if v != latest_args_dict[k]:
                raise Exception("api changed: %s \nbefore: %s \nnow: %s" \
                      %(k, latest_args_dict[k], v))
        for k, v in latest_args_dict.items():
            # if api removed, KeyError occurs, it be catched later. 
            if v != args_dict[k]:
                raise Exception("api changed: %s \nbefore: %s \nnow: %s" \
                      %(k, latest_args_dict[k], v))


if __name__ == '__main__':
    base_dir = '../python/paddle/fluid/'
    fail_list = []
    for file_name in get_all_files(base_dir):
        if file_name.split('/')[4] == 'tests':
            continue
        if file_name.split('/')[-1].endswith('__.py'):
            continue
        module_str = '.'.join(file_name.split('/')[3:])
        module_str = '.'.join(module_str.split('.')[:-1])
        print module_str
        args = parse_args()
        if args.op == 'save':
            save_api(module_str, args.dir)
        elif args.op == 'check':
            try:
                check_api(module_str, args.dir)
            except Exception as e:
                print "NOTE:", module_str, e
                print traceback.format_exc()
                fail_list.append(module_str)
    if fail_list:
        sys.exit(-1)
