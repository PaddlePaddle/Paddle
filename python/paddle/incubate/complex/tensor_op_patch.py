# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from ...fluid import framework
from . import tensor


def monkey_patch_math_complex():
    complex_methods = [
        ('__add__', _binary_creator_('__add__', "elementwise_add", False)),
        ('__radd__', _binary_creator_('__add__', "elementwise_add", True)),
        ('__sub__', _binary_creator_('__sub__', "elementwise_sub", False)),
        ('__rsub__', _binary_creator_('__rsub__', "elementwise_sub", True)),
        ('__mul__', _binary_creator_('__mul__', "elementwise_mul", False)),
        ('__rmul__', _binary_creator_('__rmul__', "elementwise_mul", True)),
        ('__truediv__', _binary_creator_('__truediv__', "elementwise_div",
                                         False)),
        ('__rtruediv__', _binary_creator_('__rtruediv__', "elementwise_div",
                                          True)),
        ('__matmul__', _binary_creator_('__matmul__', "matmul", False)),
        ('__rmatmul__', _binary_creator_('__rmatmul__', "matmul", True)),
    ]

    for method in complex_methods:
        method_name = method[0]
        method_impl = method[1]
        if method_impl:
            setattr(framework.ComplexVariable, method_name, method_impl)

    for method in tensor.__all__:
        method_impl = getattr(tensor, method)
        if method_impl:
            setattr(framework.ComplexVariable, method, method_impl)


# for binary operator such as elementwise, compare
def _binary_creator_(method_name, op_type, reverse=False):
    def __impl__(self, other_var):
        if reverse:
            tmp = self
            self = other_var
            other_var = tmp

        math_op = getattr(tensor, op_type)
        return math_op(self, other_var)

    __impl__.__name__ = method_name
    return __impl__
