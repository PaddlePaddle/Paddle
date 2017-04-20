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

import functools
import inspect
from .attrs import ParamAttr
from .activations import TanhActivation
from paddle.trainer.config_parser import *

__all__ = [
    'wrap_name_default', 'wrap_param_attr_default', 'wrap_bias_attr_default',
    'wrap_act_default', 'wrap_param_default'
]


def __default_not_set_callback__(kwargs, name):
    return name not in kwargs or kwargs[name] is None


def wrap_param_default(param_names=None,
                       default_factory=None,
                       not_set_callback=__default_not_set_callback__):
    assert param_names is not None
    assert isinstance(param_names, list) or isinstance(param_names, tuple)
    for each_param_name in param_names:
        assert isinstance(each_param_name, basestring)

    def __impl__(func):
        @functools.wraps(func)
        def __wrapper__(*args, **kwargs):
            if len(args) != 0:
                argspec = inspect.getargspec(func)
                num_positional = len(argspec.args)
                if argspec.defaults:
                    num_positional -= len(argspec.defaults)
                if not argspec.varargs and len(args) > num_positional:
                    logger.fatal(
                        "Must use keyword arguments for non-positional args")
            for name in param_names:
                if not_set_callback(kwargs, name):  # Not set
                    kwargs[name] = default_factory(func)
            return func(*args, **kwargs)

        if hasattr(func, 'argspec'):
            __wrapper__.argspec = func.argspec
        else:
            __wrapper__.argspec = inspect.getargspec(func)
        return __wrapper__

    return __impl__


class DefaultNameFactory(object):
    def __init__(self, name_prefix):
        self.__counter__ = 0
        self.__name_prefix__ = name_prefix

    def __call__(self, func):
        if self.__name_prefix__ is None:
            self.__name_prefix__ = func.__name__
        tmp = "__%s_%d__" % (self.__name_prefix__, self.__counter__)
        self.__check_name__(tmp)
        self.__counter__ += 1
        return tmp

    def __check_name__(self, nm):
        """
        @TODO(yuyang18): Implement it!
        @param nm:
        @return:
        """
        pass

    def reset(self):
        self.__counter__ = 0


_name_factories = []


def reset_hook():
    for factory in _name_factories:
        factory.reset()


register_parse_config_hook(reset_hook)


def wrap_name_default(name_prefix=None, name_param="name"):
    """
    Decorator to set "name" arguments default to "{name_prefix}_{invoke_count}".

    ..  code:: python

        @wrap_name_default("some_name")
        def func(name=None):
            print name      # name will never be None. If name is not set,
                            # name will be "some_name_%d"

    :param name_prefix: name prefix. wrapped function's __name__ if None.
    :type name_prefix: basestring
    :return: a decorator to set default name
    :rtype: callable
    """
    factory = DefaultNameFactory(name_prefix)
    _name_factories.append(factory)
    return wrap_param_default([name_param], factory)


def wrap_param_attr_default(param_names=None, default_factory=None):
    """
    Setting Default Parameter Attributes Decorator.

    :param default_factory:
    :param param_names: Parameter Attribute's Names, list of string
    :type param_names: list
    :return: decorator
    """
    if param_names is None:
        param_names = ['param_attr']
    if default_factory is None:
        default_factory = lambda _: ParamAttr()

    return wrap_param_default(param_names, default_factory)


def wrap_bias_attr_default(param_names=None,
                           default_factory=None,
                           has_bias=True):
    if param_names is None:
        param_names = ['bias_attr']
    if default_factory is None:
        default_factory = lambda _: ParamAttr(initial_std=0., initial_mean=0.)

    def __bias_attr_not_set__(kwargs, name):
        if has_bias:
            return name not in kwargs or kwargs[name] is None or \
                   kwargs[name] == True
        else:
            return name in kwargs and kwargs[name] == True

    return wrap_param_default(param_names, default_factory,
                              __bias_attr_not_set__)


def wrap_act_default(param_names=None, act=None):
    if param_names is None:
        param_names = ["act"]

    if act is None:
        act = TanhActivation()

    return wrap_param_default(param_names, lambda _: act)
