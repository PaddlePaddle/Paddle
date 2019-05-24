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

from __future__ import print_function

import collections
from .wrapped_decorator import signature_safe_contextmanager
import six
import sys

__all__ = ['generate', 'switch', 'guard']


class UniqueNameGenerator(object):
    """
    Generate unique name with prefix.

    Args:
        prefix(str): The generated name prefix. All generated name will be
                     started with this prefix.
    """

    def __init__(self, prefix=None):
        self.ids = collections.defaultdict(int)
        if prefix is None:
            prefix = ""
        self.prefix = prefix

    def __call__(self, key):
        """
        Generate unique names with prefix

        Args:
            key(str): The key of return string.

        Returns(str): A unique string with the prefix
        """
        tmp = self.ids[key]
        self.ids[key] += 1
        return self.prefix + "_".join([key, str(tmp)])


generator = UniqueNameGenerator()


def generate(key):
    return generator(key)


# FIXME(zjl): The previous naming rule in static graph would
# cause memory leak in dygraph mode. It is because the previous
# nameing rule would use `conv_0.tmp` as the key, and in dygraph
# mode, `conv_i` increases as batch increases. Thus, keys would
# increase in a way like `conv_0.tmp`, `conv_1.tmp`, .... 
# Not find a better way to fix this bug in dygraph mode. In TF,
# variable name is meaningless in eager execution mode, and in
# PyTorch, there is no variable name at all. Maybe we should
# discard variable name in dygraph mode.
#
# Another concern is that save/load inference. Usually, user
# would save model in static graph mode, and load it in dygraph
# mode. Therefore, we keep the variable name of Parameter currently.
# 
# Please fix me if a better method is found.        
def generate_with_ignorable_key(key):
    from .framework import in_dygraph_mode
    if in_dygraph_mode():
        key = "tmp"

    return generator(key)


def switch(new_generator=None):
    global generator
    old = generator
    if new_generator is None:
        generator = UniqueNameGenerator()
    else:
        generator = new_generator
    return old


@signature_safe_contextmanager
def guard(new_generator=None):
    if isinstance(new_generator, six.string_types):
        new_generator = UniqueNameGenerator(new_generator)
    elif isinstance(new_generator, six.binary_type):
        new_generator = UniqueNameGenerator(new_generator.decode())
    old = switch(new_generator)
    yield
    switch(old)
