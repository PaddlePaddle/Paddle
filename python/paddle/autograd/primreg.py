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
    __slots__ = ['name', 'tab']

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
_orig2prim = Registry('orig2prim')
_prim2orig = Registry('prim2orig')
_primop_jvp = Registry('primop_jvp')
_primop_transpose = Registry('primop_transpose')
_primop_position_argnames = Registry('primop_position_argnames')


def REGISTER_FN(op_type, *position_argnames):
    """Decorator for registering the Python function for a primitive op."""

    assert isinstance(op_type, str)

    _primop_position_argnames.register(op_type, position_argnames)

    def wrapper(f):
        _primop_fn.register(op_type, f)
        return f

    return wrapper
