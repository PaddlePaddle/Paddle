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
import typing

import paddle
from paddle.fluid import framework as framework


class PrimOption:
    def __init__(self):
        self.enable_prim = False

    def get_status(self):
        return self.enable_prim

    def set_status(self, flag):
        self.enable_prim = flag


prim_option = PrimOption()


@framework.static_only
def prim_enabled():
    """
    Note:
        **ONLY available in the static mode.**

    Shows whether the automatic differentiation mechanism based on
    automatic differential basic operators is ON. Defaults to OFF.

    Returns:
        flag(bool): Whether the automatic differentiation mechanism based on automatic differential basic operators is ON.

    Examples:

        .. code-block:: python

            import paddle
            from paddle.incubate.autograd import enable_prim, disable_prim, prim_enabled

            paddle.enable_static()
            enable_prim()

            print(prim_enabled()) # True

            disable_prim()

            print(prim_enabled()) # False
    """
    return prim_option.get_status()


@framework.static_only
def enable_prim():
    """
    Note:
        **ONLY available in the static mode.**

    Turns ON automatic differentiation mechanism based on automatic
    differential basic operators.

    Examples:

        .. code-block:: python

            import paddle
            from paddle.incubate.autograd import enable_prim, prim_enabled

            paddle.enable_static()
            enable_prim()

            print(prim_enabled()) # True
    """
    prim_option.set_status(True)


@framework.static_only
def disable_prim():
    """
    Note:
        **ONLY available in the static mode.**

    Turns OFF automatic differentiation mechanism based on automatic
    differential basic operators.

    Examples:

        .. code-block:: python

            import paddle
            from paddle.incubate.autograd import enable_prim, disable_prim, prim_enabled

            paddle.enable_static()
            enable_prim()

            print(prim_enabled()) # True

            disable_prim()

            print(prim_enabled()) # False
    """
    prim_option.set_status(False)


INT_DTYPE_2_STRING = {
    int(0): 'bool',
    int(1): 'int16',
    int(2): 'int32',
    int(3): 'int64',
    int(4): 'float16',
    int(5): 'float32',
    int(6): 'float64',
    int(20): 'uint8',
    int(21): 'int8',
    int(23): 'complex64',
    int(24): 'complex128',
}


def get_var_block(block, names):
    assert isinstance(names, list)
    if len(names) == 0:
        return None
    elif len(names) == 1:
        return block.var(names[0])
    else:
        return [block.var(name) for name in names]


def get_input_var_list(op):
    if op.input_names is None:
        return []
    else:
        return [
            get_var_block(op.block, op.input(n)) for n in sorted(op.input_names)
        ]


def get_output_var_list(op):
    if op.output_names is None:
        return []
    else:
        return [
            get_var_block(op.block, op.output(n))
            for n in sorted(op.output_names)
        ]


def flatten(inp):
    if inp is None or isinstance(inp, paddle.fluid.framework.Variable):
        return [inp]
    flattened = []
    for part in inp:
        flattened += flatten(part)
    return flattened


def flatten_and_remove_none(inp):
    flattened = flatten(inp)
    return [var for var in flattened if var is not None]


def as_tensors(xs):
    if isinstance(xs, framework.Variable):
        return (xs,)
    elif isinstance(xs, typing.Sequence):
        return tuple(xs)
    else:
        return xs
