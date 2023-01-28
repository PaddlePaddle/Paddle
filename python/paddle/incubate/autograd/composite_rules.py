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

# This file contains composite rules of nonbasic operations. There are some notes:
# 1. When define composite rule of some op, you can only use primitive ops defined in primitives.py.
# 2. The name and args of target op must be corresponding with standard description of op in
#    ops.yaml or legacy_ops.yaml.

from .primitives import *  # noqa: F403
from .primreg import REGISTER_COMPOSITE, lookup_composite


def _composite(op, *args):
    _lowerrule = lookup_composite(op.type)
    return _lowerrule(op, *args)


@REGISTER_COMPOSITE('softmax')
def softmax_composite(x, axis):
    """define composite rule of op softmax"""
    max_temp = max(x, axis, keepdim=True)
    max_temp.stop_gradient = True
    molecular = exp(x - max_temp)
    denominator = sum(molecular, axis=axis, keepdim=True)
    res = divide(molecular, denominator)
    return res
