# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved
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
import inspect


class Registry:
    """A general registry object."""

    __slots__ = ['name', 'rules']

    def __init__(self, name):
        self.name = name
        self.rules = {}

    def register(self, op_type, rule):
        assert isinstance(op_type, str)
        assert inspect.isfunction(rule)
        assert (
            op_type not in self.rules
        ), f'name "{op_type}" should not be registered before.'
        self.rules[op_type] = rule

    def lookup(self, op_type):
        return self.rules.get(op_type)


_decomposition_ops = Registry('decomposition')


def register_decomp(op_type):
    """
    Decorator for registering the lower function for an original op into sequence of primitive ops.

    Args:
        op_type(str): The op name

    Returns:
        wrapper: Inner wrapper function

    Examples:
        .. code-block:: python

            >>> from paddle.decomposition import register
            >>> @register.register_decomp('softmax')
            >>> def softmax(x, axis):
            ...     molecular = exp(x)
            ...     denominator = broadcast_to(sum(molecular, axis=axis, keepdim=True), x.shape)
            ...     res = divide(molecular, denominator)
            ...     return res
    """
    if not isinstance(op_type, str):
        raise TypeError(f'op_type must be str, but got {type(op_type)}.')

    def wrapper(f):
        _decomposition_ops.register(op_type, f)
        return f

    return wrapper


def get_decomp_rule(op_type):
    _lowerrule = _decomposition_ops.lookup(op_type)
    return _lowerrule
