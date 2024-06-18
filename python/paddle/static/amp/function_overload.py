# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

# The implementation refers to https://arpitbhayani.me/blogs/function-overloading.
# Note: it is customed for paddle.static.amp.decorate function.

import inspect
import logging
from enum import Enum

from paddle.base.log_helper import get_logger

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s'
)


class FunctionType(Enum):
    FP16_ONLY = 0
    COMMON = 1


class Function:
    """
    Function is a wrap over standard python function
    An instance of this Function class is also callable
    just like the python function that it wrapped.
    When the instance is "called" like a function it fetches
    the function to be invoked from the virtual namespace and then
    invokes the same.
    """

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args, **kwargs):
        """
        Overriding the __call__ function which makes the
        instance callable.
        """
        # fetching the function to be invoked from the virtual namespace
        # through the arguments.
        fn = Namespace.get_instance().get(*args, **kwargs)
        # invoking the wrapped function and returning the value.
        return fn(*args, **kwargs)


class Namespace:
    """
    Namespace is the singleton class that is responsible
    for holding all the functions.
    """

    __instance = None

    def __init__(self):
        if self.__instance is None:
            self.function_map = {}
            Namespace.__instance = self
        else:
            raise Exception("cannot instantiate Namespace again.")

    @staticmethod
    def get_instance():
        if Namespace.__instance is None:
            Namespace()
        return Namespace.__instance

    def register(self, fn, key):
        """
        Register the function in the virtual namespace and return
        an instance of callable Function that wraps the function fn.

        Args:
            fn (function): the native python function handle.
            key (FunctionType): the specified type.
        """
        assert isinstance(
            key, FunctionType
        ), f"The type of  key is expected to be FunctionType, but received {type(key)}."
        func = Function(fn)
        self.function_map[key] = fn
        return func

    def get(self, *args, **kwargs):
        """
        Get the matching function from the virtual namespace according to the actual arguments.
        Return None if it did not find any matching function.
        """
        _logger.debug(f"get function: args={args}, kwargs={kwargs}")
        satisfied_function_keys = set(self.function_map.keys())
        num_actual_args = len(args) + len(kwargs)
        for func_key in self.function_map.keys():
            if func_key not in satisfied_function_keys:
                continue
            fn = self.function_map[func_key]
            specs = inspect.getfullargspec(fn)
            if len(specs) < len(args) + len(kwargs):
                # Remove the not satisfied function according to the number of actual arguments.
                _logger.debug(
                    f"fn={fn} (key={func_key}) is not satisfied and removed."
                )
                satisfied_function_keys.remove(func_key)
                continue
            if len(kwargs) > 0:
                # Remove the not satisfied function according to argument keys in kwargs.
                for arg_name, value in kwargs.items():
                    if arg_name not in specs.args:
                        _logger.debug(
                            f"fn={fn} (key={func_key}) is not satisfied and removed."
                        )
                        satisfied_function_keys.remove(func_key)
                        break
        if len(satisfied_function_keys) == 1:
            key = list(satisfied_function_keys)[0]
        elif len(args) >= 3 and isinstance(args[2], float):
            key = FunctionType.FP16_ONLY
        else:
            key = FunctionType.COMMON
        return self.function_map.get(key)


def overload(key):
    """overload is the decorator that wraps the function
    and returns a callable object of type Function.
    """

    def decorator(fn):
        return Namespace.get_instance().register(fn, key)

    return decorator
