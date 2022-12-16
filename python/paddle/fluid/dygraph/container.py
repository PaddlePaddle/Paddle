# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from .layers import Layer

__all__ = [
    'Sequential',
]


class Sequential(Layer):
    """Sequential container.
    Sub layers will be added to this container in the order of argument in the constructor.
    The argument passed to the constructor can be iterable Layers or iterable name Layer pairs.

    Parameters:
        layers(Layer|list|tuple): Layer or list/tuple of iterable name Layer pair.

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            data = np.random.uniform(-1, 1, [30, 10]).astype('float32')
            data = paddle.to_tensor(data)
            # create Sequential with iterable Layers
            model1 = paddle.nn.Sequential(
                paddle.nn.Linear(10, 1), paddle.nn.Linear(1, 2)
            )
            model1[0]  # access the first layer
            res1 = model1(data)  # sequential execution

            # create Sequential with name Layer pairs
            model2 = paddle.nn.Sequential(
                ('l1', paddle.nn.Linear(10, 2)),
                ('l2', paddle.nn.Linear(2, 3))
            )
            model2['l1']  # access l1 layer
            model2.add_sublayer('l3', paddle.nn.Linear(3, 3))  # add sublayer
            res2 = model2(data)  # sequential execution

    """

    def __init__(self, *layers):
        super().__init__()
        if len(layers) > 0 and isinstance(layers[0], (list, tuple)):
            for name, layer in layers:
                self.add_sublayer(name, layer)
        else:
            for idx, layer in enumerate(layers):
                self.add_sublayer(str(idx), layer)

    def __getitem__(self, name):
        if isinstance(name, slice):
            return self.__class__(*(list(self._sub_layers.values())[name]))
        elif isinstance(name, str):
            return self._sub_layers[name]
        else:
            if name >= len(self._sub_layers):
                raise IndexError('index {} is out of range'.format(name))
            elif name < 0 and name >= -len(self._sub_layers):
                name += len(self._sub_layers)
            elif name < -len(self._sub_layers):
                raise IndexError('index {} is out of range'.format(name))
            return list(self._sub_layers.values())[name]

    def __setitem__(self, name, layer):
        assert isinstance(layer, Layer)
        setattr(self, str(name), layer)

    def __delitem__(self, name):
        name = str(name)
        assert name in self._sub_layers
        del self._sub_layers[name]

    def __len__(self):
        return len(self._sub_layers)

    def forward(self, input):
        for layer in self._sub_layers.values():
            input = layer(input)
        return input
