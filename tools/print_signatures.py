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
    ./print_signature  "paddle.fluid" > signature.txt
"""
from __future__ import print_function

import importlib
import inspect
import collections
import sys
import pydoc
import hashlib
import platform
import functools

member_dict = collections.OrderedDict()

visited_modules = set()


def md5(doc):
    try:
        hashinst = hashlib.md5()
        if platform.python_version()[0] == "2":
            hashinst.update(str(doc))
        else:
            hashinst.update(str(doc).encode('utf-8'))
        md5sum = hashinst.hexdigest()
    except UnicodeDecodeError as e:
        md5sum = None
        print(
            "Error({}) occurred when `md5({})`, discard it.".format(
                str(e), doc),
            file=sys.stderr)

    return md5sum


def get_functools_partial_spec(func):
    func_str = func.func.__name__
    args = func.args
    keywords = func.keywords
    return '{}(args={}, keywords={})'.format(func_str, args, keywords)


def format_spec(spec):
    args = spec.args
    varargs = spec.varargs
    keywords = spec.keywords
    defaults = spec.defaults
    if defaults is not None:
        defaults = list(defaults)
        for idx, item in enumerate(defaults):
            if not isinstance(item, functools.partial):
                continue

            defaults[idx] = get_functools_partial_spec(item)

        defaults = tuple(defaults)

    return 'ArgSpec(args={}, varargs={}, keywords={}, defaults={})'.format(
        args, varargs, keywords, defaults)


def queue_dict(member, cur_name):
    if cur_name != 'paddle':
        try:
            eval(cur_name)
        except (AttributeError, NameError, SyntaxError) as e:
            print(
                "Error({}) occurred when `eval({})`, discard it.".format(
                    str(e), cur_name),
                file=sys.stderr)
            return

    if (inspect.isclass(member) or inspect.isfunction(member) or
            inspect.ismethod(member)) and hasattr(
                member, '__module__') and hasattr(member, '__name__'):
        args = member.__module__ + "." + member.__name__
        try:
            eval(args)
        except (AttributeError, NameError, SyntaxError) as e:
            print(
                "Error({}) occurred when `eval({})`, discard it for {}.".format(
                    str(e), args, cur_name),
                file=sys.stderr)
            return
    else:
        try:
            args = inspect.getargspec(member)
            has_type_error = False
        except TypeError:  # special for PyBind method
            args = "  ".join([
                line.strip() for line in pydoc.render_doc(member).split('\n')
                if "->" in line
            ])
            has_type_error = True

        if not has_type_error:
            args = format_spec(args)

    doc_md5 = md5(member.__doc__)
    member_dict[cur_name] = "({}, ('document', '{}'))".format(args, doc_md5)


def visit_member(parent_name, member, member_name=None):
    if member_name:
        cur_name = ".".join([parent_name, member_name])
    else:
        cur_name = ".".join([parent_name, member.__name__])
    if inspect.isclass(member):
        queue_dict(member, cur_name)
        for name, value in inspect.getmembers(member):
            if hasattr(value, '__name__') and not name.startswith("_"):
                visit_member(cur_name, value)
    elif inspect.ismethoddescriptor(member):
        return
    elif inspect.isbuiltin(member):
        return
    elif callable(member):
        queue_dict(member, cur_name)
    elif inspect.isgetsetdescriptor(member):
        return
    else:
        raise RuntimeError("Unsupported generate signature of member, type {0}".
                           format(str(type(member))))


def is_primitive(instance):
    int_types = (int, long) if platform.python_version()[0] == "2" else (int, )
    pritimitive_types = int_types + (float, str)
    if isinstance(instance, pritimitive_types):
        return True
    elif isinstance(instance, (list, tuple, set)):
        for obj in instance:
            if not is_primitive(obj):
                return False

        return True
    else:
        return False


def visit_all_module(mod):
    mod_name = mod.__name__
    if mod_name != 'paddle' and not mod_name.startswith('paddle.'):
        return

    if mod_name.startswith('paddle.fluid.core'):
        return

    if mod in visited_modules:
        return

    visited_modules.add(mod)
    if hasattr(mod, "__all__"):
        member_names = (name for name in mod.__all__
                        if not name.startswith("_"))
    elif mod_name == 'paddle':
        member_names = dir(mod)
    else:
        return
    for member_name in member_names:
        instance = getattr(mod, member_name, None)
        if instance is None:
            continue

        if is_primitive(instance):
            continue

        if not hasattr(instance, "__name__"):
            continue

        if inspect.ismodule(instance):
            visit_all_module(instance)
        else:
            if member_name != instance.__name__:
                print(
                    "Found alias API, alias name is: {}, original name is: {}".
                    format(member_name, instance.__name__),
                    file=sys.stderr)
                visit_member(mod.__name__, instance, member_name)
            else:
                visit_member(mod.__name__, instance)


if __name__ == '__main__':
    import paddle
    modules = sys.argv[1].split(",")
    for m in modules:
        visit_all_module(importlib.import_module(m))

    for name in member_dict:
        print(name, member_dict[name])
