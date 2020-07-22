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

import os
import inspect
import numpy as np

from collections import OrderedDict
from paddle import fluid
from paddle.fluid.framework import Variable
from paddle.fluid.executor import global_scope

__all__ = ['uncombined_weight_to_state_dict']


def uncombined_weight_to_state_dict(weight_dir):
    """
    Convert uncombined weight to state_dict

    Args:
        weight_dir (str): weight direcotory path.

    Returns:
        OrderDict: weight dict.
    """

    def _get_all_params_name(dir):
        params_name = []
        dir = os.path.expanduser(dir)

        dir_len = len(dir)
        for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root[dir_len:], fname)
                params_name.append(path)

        return params_name

    class Load(fluid.dygraph.Layer):
        def __init__(self):
            super(Load, self).__init__()

        def forward(self, filename):
            weight = self.create_parameter(
                shape=[1],
                dtype='float32',
                default_initializer=fluid.initializer.ConstantInitializer(0.0))
            self._helper.append_op(
                type='load',
                inputs={},
                outputs={'Out': [weight]},
                attrs={'file_path': filename})
            return weight

    params_name_list = _get_all_params_name(weight_dir)
    if not fluid.in_dygraph_mode():
        dygraph_enabled = False
        fluid.enable_imperative()
    else:
        dygraph_enabled = True

    load = Load()
    state_dict = OrderedDict()

    for param_name in params_name_list:
        param_path = os.path.join(weight_dir, param_name)
        weight = load(param_path)
        try:
            weight = weight.numpy()
        except Exception as e:
            print(e)

        state_dict[param_name] = weight

    if not dygraph_enabled:
        fluid.disable_imperative()

    return state_dict


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
