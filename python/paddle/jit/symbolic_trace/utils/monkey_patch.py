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

from .utils import no_eval_frame


# The MoneyPatch module adds methods to a class.
def proxy_tensor_method_builder(method_name):
    @no_eval_frame
    def __impl__(self, other):
        return self.call_method(method_name, self, other)

    return __impl__


def do_monkey_patch(cls, patch_names, method_builder):
    for method_name in patch_names:
        setattr(cls, method_name, method_builder(method_name))


binary_operator_methods = [
    '__add__',
    '__sub__',
    '__rsub__',
    '__radd__',
    '__mul__',
    '__rmul__',
    '__gt__',
    '__xor__',
    '__or__',
    '__and__',
    '__mod__',
    '__matmul__',
    '__pow__',
    '__floordiv__',
    '__truediv__',
    '__lshift__',
    '__rshift__',
]

unary_operator_methods = [
    '__invert__',
    '__neg__',
    '__pos__',
]
