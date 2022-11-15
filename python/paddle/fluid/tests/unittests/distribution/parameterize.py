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
import collections
import functools
import inspect
import re
import sys

from unittest import SkipTest

import numpy as np
import config

TEST_CASE_NAME = 'suffix'


def xrand(shape=(10, 10, 10), dtype=config.DEFAULT_DTYPE, min=1.0, max=10.0):
    return (np.random.rand(*shape).astype(dtype)) * (max - min) + min


def place(devices, key='place'):
    def decorate(cls):
        module = sys.modules[cls.__module__].__dict__
        raw_classes = {
            k: v for k, v in module.items() if k.startswith(cls.__name__)
        }

        for raw_name, raw_cls in raw_classes.items():
            for d in devices:
                test_cls = dict(raw_cls.__dict__)
                test_cls.update({key: d})
                new_name = raw_name + '.' + d.__class__.__name__
                module[new_name] = type(new_name, (raw_cls,), test_cls)
            del module[raw_name]
        return cls

    return decorate


def parameterize_cls(fields, values=None):
    fields = [fields] if isinstance(fields, str) else fields
    params = [dict(zip(fields, vals)) for vals in values]

    def decorate(cls):
        test_cls_module = sys.modules[cls.__module__].__dict__
        for k, v in enumerate(params):
            test_cls = dict(cls.__dict__)
            test_cls.update(v)
            name = cls.__name__ + str(k)
            name = name + '.' + v.get('suffix') if v.get('suffix') else name

            test_cls_module[name] = type(name, (cls,), test_cls)

        for m in list(cls.__dict__):
            if m.startswith("test"):
                delattr(cls, m)
        return cls

    return decorate


def parameterize_func(
    input, name_func=None, doc_func=None, skip_on_empty=False
):
    name_func = name_func or default_name_func

    def wrapper(f, instance=None):
        frame_locals = inspect.currentframe().f_back.f_locals

        parameters = input_as_callable(input)()

        if not parameters:
            if not skip_on_empty:
                raise ValueError(
                    "Parameters iterable is empty (hint: use "
                    "`parameterized.expand([], skip_on_empty=True)` to skip "
                    "this test when the input is empty)"
                )
            return functools.wraps(f)(skip_on_empty_helper)

        digits = len(str(len(parameters) - 1))
        for num, p in enumerate(parameters):
            name = name_func(
                f, "{num:0>{digits}}".format(digits=digits, num=num), p
            )
            # If the original function has patches applied by 'mock.patch',
            # re-construct all patches on the just former decoration layer
            # of param_as_standalone_func so as not to share
            # patch objects between new functions
            nf = reapply_patches_if_need(f)
            frame_locals[name] = param_as_standalone_func(p, nf, name)
            frame_locals[name].__doc__ = f.__doc__

        # Delete original patches to prevent new function from evaluating
        # original patching object as well as re-constrfucted patches.
        delete_patches_if_need(f)

        f.__test__ = False

    return wrapper


def reapply_patches_if_need(func):
    def dummy_wrapper(orgfunc):
        @functools.wraps(orgfunc)
        def dummy_func(*args, **kwargs):
            return orgfunc(*args, **kwargs)

        return dummy_func

    if hasattr(func, 'patchings'):
        func = dummy_wrapper(func)
        tmp_patchings = func.patchings
        delattr(func, 'patchings')
        for patch_obj in tmp_patchings:
            func = patch_obj.decorate_callable(func)
    return func


def delete_patches_if_need(func):
    if hasattr(func, 'patchings'):
        func.patchings[:] = []


def default_name_func(func, num, p):
    base_name = func.__name__
    name_suffix = "_%s" % (num,)

    if len(p.args) > 0 and isinstance(p.args[0], str):
        name_suffix += "_" + to_safe_name(p.args[0])
    return base_name + name_suffix


def param_as_standalone_func(p, func, name):
    @functools.wraps(func)
    def standalone_func(*a):
        return func(*(a + p.args), **p.kwargs)

    standalone_func.__name__ = name

    # place_as is used by py.test to determine what source file should be
    # used for this test.
    standalone_func.place_as = func

    # Remove __wrapped__ because py.test will try to look at __wrapped__
    # to determine which parameters should be used with this test case,
    # and obviously we don't need it to do any parameterization.
    try:
        del standalone_func.__wrapped__
    except AttributeError:
        pass
    return standalone_func


def input_as_callable(input):
    if callable(input):
        return lambda: check_input_values(input())
    input_values = check_input_values(input)
    return lambda: input_values


def check_input_values(input_values):
    if not isinstance(input_values, list):
        input_values = list(input_values)
    return [param.from_decorator(p) for p in input_values]


def skip_on_empty_helper(*a, **kw):
    raise SkipTest("parameterized input is empty")


_param = collections.namedtuple("param", "args kwargs")


class param(_param):
    def __new__(cls, *args, **kwargs):
        return _param.__new__(cls, args, kwargs)

    @classmethod
    def explicit(cls, args=None, kwargs=None):
        """Creates a ``param`` by explicitly specifying ``args`` and
        ``kwargs``::
            >>> param.explicit([1,2,3])
            param(*(1, 2, 3))
            >>> param.explicit(kwargs={"foo": 42})
            param(*(), **{"foo": "42"})
        """
        args = args or ()
        kwargs = kwargs or {}
        return cls(*args, **kwargs)

    @classmethod
    def from_decorator(cls, args):
        """Returns an instance of ``param()`` for ``@parameterized`` argument
        ``args``::
            >>> param.from_decorator((42, ))
            param(args=(42, ), kwargs={})
            >>> param.from_decorator("foo")
            param(args=("foo", ), kwargs={})
        """
        if isinstance(args, param):
            return args
        elif isinstance(args, str):
            args = (args,)
        try:
            return cls(*args)
        except TypeError as e:
            if "after * must be" not in str(e):
                raise
            raise TypeError(
                "Parameters must be tuples, but %r is not (hint: use '(%r, )')"
                % (args, args),
            )

    def __repr__(self):
        return "param(*%r, **%r)" % self


def to_safe_name(s):
    return str(re.sub("[^a-zA-Z0-9_]+", "_", s))


# alias
parameterize = parameterize_func
param_cls = parameterize_cls
param_func = parameterize_func
