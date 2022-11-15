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

from numpy import prod

_FLOPS_COMPUTE_FUNC_MAP = {}


def flops(op_type: str, input_shapes: tuple, **attrs) -> int:
    """
    count flops for operation.

    Args:
        op_type (str): the type of operation.
        input_shapes (tuple): the shapes of inputs.
        attrs (dict): the attributes of the operation.

    Returns:
        the total flops of the operation.
    """

    if op_type not in _FLOPS_COMPUTE_FUNC_MAP:
        return 0
    else:
        func = _FLOPS_COMPUTE_FUNC_MAP[op_type]
        return func(input_shapes, **attrs)


def register_flops(op_type):
    """
    register flops computation function for operation.
    """

    def register(func):
        global _FLOPS_COMPUTE_FUNC_MAP
        _FLOPS_COMPUTE_FUNC_MAP[op_type] = func
        return func

    return register


@register_flops("dropout")
def _dropout_flops(input_shapes, **attrs):
    return 0


@register_flops("relu")
def _relu_flops(input_shapes, **attrs):
    return prod(input_shapes[0])
