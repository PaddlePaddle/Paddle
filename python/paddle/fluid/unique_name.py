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
    """
    Generate unique name with prefix key.

    Args:
        key(str): The generated name prefix. All generated name will be 
                  started with this prefix.

    Returns: 
        str: A unique string with the prefix key.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            name1 = fluid.unique_name.generate('fc')
            name2 = fluid.unique_name.generate('fc')
            # The result is fc_0, fc_1
            print name1, name2 
    """
    return generator(key)


# FIXME(zjl): The previous naming rule in static graph would
# cause memory leak in dygraph mode. It is because the previous
# naming rule would use `conv_0.tmp` as the key, and in dygraph
# mode, `conv_i` increases as batch increases. Thus, keys would
# increase in a way like `conv_0.tmp`, `conv_1.tmp`, .... 
# Not find a better way to fix this bug in dygraph mode. In TF,
# variable name is meaningless in eager execution mode, and in
# PyTorch, there is no variable name at all. Maybe we should
# discard variable name in dygraph mode.
#
# Another concern is that save/load interfaces. Usually, user
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
    """
    Switch the Global namespace to a new namespace.

    Args:
        new_generator(None|UniqueNameGenerator): A new UniqueNameGenerator.

    Returns: 
        UniqueNameGenerator: The previous UniqueNameGenerator.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            name1 = fluid.unique_name.generate('fc')
            name2 = fluid.unique_name.generate('fc')
            # The result is fc_0, fc_1
            print name1, name2 

            fluid.unique_name.switch()
            name2 = fluid.unique_name.generate('fc')
            # The result is fc_0
            print name2
    """
    global generator
    old = generator
    if new_generator is None:
        generator = UniqueNameGenerator()
    else:
        generator = new_generator
    return old


@signature_safe_contextmanager
def guard(new_generator=None):
    """
    Change the global namespace with `with` statement.
    
    Args:
        new_generator(None|str|bytes): New name of global namespace.
            Note that str in Python2 was spilted into str and bytes in Python3, 
            so here are two types. Default is None.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            with fluid.unique_name.guard():
              name_1 = fluid.unique_name.generate('fc')
            with fluid.unique_name.guard():
              name_2 = fluid.unique_name.generate('fc')
            # The result is fc_0, fc_0
            print name_1, name_2

            with fluid.unique_name.guard('A'):
              name_1 = fluid.unique_name.generate('fc')
            with fluid.unique_name.guard('B'):
              name_2 = fluid.unique_name.generate('fc')
            # The result is Afc_0, Bfc_0
            print name_1, name_2
    """
    if isinstance(new_generator, six.string_types):
        new_generator = UniqueNameGenerator(new_generator)
    elif isinstance(new_generator, six.binary_type):
        new_generator = UniqueNameGenerator(new_generator.decode())
    old = switch(new_generator)
    yield
    switch(old)
