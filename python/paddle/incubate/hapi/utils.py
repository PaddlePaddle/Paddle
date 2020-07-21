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

import inspect
import numpy as np

from paddle import fluid
from paddle.fluid.framework import Variable
from paddle.fluid.executor import global_scope


def to_list(value):
    if value is None:
        return value
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def to_numpy(var):
    assert isinstance(var, (Variable, fluid.core.VarBase)), "not a variable"
    if isinstance(var, fluid.core.VarBase):
        return var.numpy()
    t = global_scope().find_var(var.name).get_tensor()
    return np.array(t)


def flatten_list(l):
    assert isinstance(l, list), "not a list"
    outl = []
    splits = []
    for sl in l:
        assert isinstance(sl, list), "sub content not a list"
        splits.append(len(sl))
        outl += sl
    return outl, splits


def restore_flatten_list(l, splits):
    outl = []
    for split in splits:
        assert len(l) >= split, "list length invalid"
        sl, l = l[:split], l[split:]
        outl.append(sl)
    return outl


def extract_args(func):
    if hasattr(inspect, 'getfullargspec'):
        return inspect.getfullargspec(func)[0]
    else:
        return inspect.getargspec(func)[0]
