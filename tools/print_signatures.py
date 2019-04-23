# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
Print all signature of a python module in alphabet order.

Usage:
    ./print_signature  "paddle.fluid,paddle.reader" > signature.txt
"""
from __future__ import print_function

import importlib
import inspect
import collections
import sys
import pydoc
import hashlib

member_dict = collections.OrderedDict()

experimental_namespace = {"paddle.fluid.dygraph"}


def md5(doc):
    hash = hashlib.md5()
    hash.update(str(doc).encode('utf-8'))
    return hash.hexdigest()


def visit_member(parent_name, member):
    cur_name = ".".join([parent_name, member.__name__])
    if inspect.isclass(member):
        for name, value in inspect.getmembers(member):
            if hasattr(value, '__name__') and (not name.startswith("_") or
                                               name == "__init__"):
                visit_member(cur_name, value)
    elif callable(member):
        try:
            doc = ('document', md5(member.__doc__))
            args = inspect.getargspec(member)
            all = (args, doc)
            member_dict[cur_name] = all
        except TypeError:  # special for PyBind method
            if cur_name in check_modules_list:
                return
            member_dict[cur_name] = "  ".join([
                line.strip() for line in pydoc.render_doc(member).split('\n')
                if "->" in line
            ])
    elif inspect.isgetsetdescriptor(member):
        return
    else:
        raise RuntimeError("Unsupported generate signature of member, type {0}".
                           format(str(type(member))))


def visit_all_module(mod):
    if (mod.__name__ in experimental_namespace):
        return
    for member_name in (
            name
            for name in (mod.__all__ if hasattr(mod, "__all__") else dir(mod))
            if not name.startswith("_")):
        instance = getattr(mod, member_name, None)
        if instance is None:
            continue
        if inspect.ismodule(instance):
            visit_all_module(instance)
        else:
            visit_member(mod.__name__, instance)


check_modules_list = ["paddle.reader.ComposeNotAligned.__init__"]
modules = sys.argv[1].split(",")
for m in modules:
    visit_all_module(importlib.import_module(m))

for name in member_dict:
    print(name, member_dict[name])
