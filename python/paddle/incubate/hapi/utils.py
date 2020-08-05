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
    Convert uncombined weight which getted by using `fluid.io.save_params` or `fluid.io.save_persistables` to state_dict

    Args:
        weight_dir (str): weight direcotory path.

    Returns:
        OrderDict: weight dict.

    Examples:
        .. code-block:: python

            import os

            from paddle import fluid
            from paddle.nn import Conv2D, Pool2D, Linear, ReLU, Sequential
            from paddle.incubate.hapi.utils import uncombined_weight_to_state_dict


            class LeNetDygraph(fluid.dygraph.Layer):
                def __init__(self, num_classes=10, classifier_activation='softmax'):
                    super(LeNetDygraph, self).__init__()
                    self.num_classes = num_classes
                    self.features = Sequential(
                        Conv2D(
                            1, 6, 3, stride=1, padding=1),
                        ReLU(),
                        Pool2D(2, 'max', 2),
                        Conv2D(
                            6, 16, 5, stride=1, padding=0),
                        ReLU(),
                        Pool2D(2, 'max', 2))

                    if num_classes > 0:
                        self.fc = Sequential(
                            Linear(400, 120),
                            Linear(120, 84),
                            Linear(
                                84, 10, act=classifier_activation))

                def forward(self, inputs):
                    x = self.features(inputs)

                    if self.num_classes > 0:
                        x = fluid.layers.flatten(x, 1)
                        x = self.fc(x)
                    return x

            # save weight use fluid.io.save_params
            save_dir = 'temp'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            start_prog = fluid.Program()
            train_prog = fluid.Program()

            x = fluid.data(name='x', shape=[None, 1, 28, 28], dtype='float32')

            with fluid.program_guard(train_prog, start_prog):
                with fluid.unique_name.guard():
                    x = fluid.data(
                        name='x', shape=[None, 1, 28, 28], dtype='float32')
                    model = LeNetDygraph()
                    output = model.forward(x)

            excutor = fluid.Executor()
            excutor.run(start_prog)

            test_prog = train_prog.clone(for_test=True)

            fluid.io.save_params(excutor, save_dir, test_prog)

            # convert uncombined weight to state dict
            state_dict = uncombined_weight_to_state_dict(save_dir)

            key2key_dict = {
                'features.0.weight': 'conv2d_0.w_0',
                'features.0.bias': 'conv2d_0.b_0',
                'features.3.weight': 'conv2d_1.w_0',
                'features.3.bias': 'conv2d_1.b_0',
                'fc.0.weight': 'linear_0.w_0',
                'fc.0.bias': 'linear_0.b_0',
                'fc.1.weight': 'linear_1.w_0',
                'fc.1.bias': 'linear_1.b_0',
                'fc.2.weight': 'linear_2.w_0',
                'fc.2.bias': 'linear_2.b_0'
            }

            fluid.enable_imperative()
            dygraph_model = LeNetDygraph()

            converted_state_dict = dygraph_model.state_dict()
            for k1, k2 in key2key_dict.items():
                converted_state_dict[k1] = state_dict[k2]

            # dygraph model load state dict which converted from uncombined weight
            dygraph_model.set_dict(converted_state_dict)
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
