# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


class Registry(object):
    """ A general registry object. """

    def __init__(self, name):
        self.name = name
        self.tab = {}

    def register(self, name, value):
        assert name not in self.tab
        self.tab[name] = value
    
    def lookup(self, name):
        assert name in self.tab, f'No registry entry is found with name: {name}'
        return self.tab[name]


_primop_fn = Registry('primop_fn')
_primop_jvp = Registry('primop_jvps')
_primop_transpose = Registry('primop_vjps')


def lookup_fn(optype):
    return _primop_fn.lookup(optype)


def lookup_jvp(optype):
    return _primop_jvp.lookup(optype)


def lookup_transpose(optype):
    return _primop_transpose.lookup(optype)


def REGISTER_FN(op_type):
    """Decorator for registering the Python function for a primitive op."""

    assert isinstance(op_type, str)
    def wrapper(f):
        _primop_fn.register(op_type, f)
        return f

    return wrapper

def REGISTER_JVP(op_type):
    """Decorator for registering the JVP function for a primitive op.
    
    Usage:
    .. code-block:: python
        @RegisterJVP('add')
        def add_jvp(op, x_dot, y_dot):
            return primops.add(x_dot, y_dot)
    
    """
    assert isinstance(op_type, str)
    def wrapper(f):
        def _jvp(op, *args, **kwargs): 
            assert op.type == op_type
            return f(op, *args, **kwargs)
        _primop_jvp.register(op_type, _jvp)
        return f

    return wrapper


def REGISTER_TRANSPOSE(op_type):
    """Decorator for registering the transpose function for a primitive op
    that denotes a linear operation in the forward AD graph.
    
    Usage:
    .. code-block:: python
        @RegisterJVP('add')
        def add_transpose(op, z_bar):
            return z_bar, z_bar
    
    """
    assert isinstance(op_type, str)
    def wrapper(f):
        def _transpose(op, dot_checker, *args, **kwargs): 
            assert op.type == op_type
            return f(op, dot_checker, *args, **kwargs)
        _primop_transpose.register(op_type, _transpose)
        return f

    return wrapper