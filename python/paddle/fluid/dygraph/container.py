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

__all__ = ['Sequential']


class Sequential(Layer):
    """Sequential container.
    Sub layers will be added to this container in the order of argument in the constructor.
    The argument passed to the constructor can be iterable Layers or iterable name Layer pairs.

    Parameters:
        name_scope(str): The name of this class.
        layers(iterable): Iterable Layers or iterable name Layer pairs.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            data = np.random.uniform(-1, 1, [30, 10]).astype('float32')
            with fluid.dygraph.guard():
                data = fluid.dygraph.to_variable(data)
                # create Sequential with iterable Layers
                model1 = fluid.dygraph.Sequential('model1',
                    fluid.FC('fc1', 2),
                    fluid.FC('fc2', 3)
                )
                model1[0]  # access fc1 layer
                res1 = model1(data)  # sequential execution

                # create Sequential with name Layer pairs
                model2 = fluid.dygraph.Sequential('model2',
                    ('l1', fluid.FC('l1', 2)),
                    ('l2', fluid.FC('l2', 3))
                )
                model2['l1']  # access l1 layer
                model2.add_sublayer('l3', fluid.FC('l3', 3))  # add sublayer
                print([l.full_name() for l in model2.sublayers()])  # ['l1/FC_0', 'l2/FC_0', 'l3/FC_0']
                res2 = model2(data)  # sequential execution

    """

    def __init__(self, name_scope, *layers):
        super(Sequential, self).__init__(name_scope)
        if len(layers) > 0 and isinstance(layers[0], tuple):
            for name, layer in layers:
                self.add_sublayer(name, layer)
        else:
            for idx, layer in enumerate(layers):
                self.add_sublayer(str(idx), layer)

    def __getitem__(self, name):
        return self._sub_layers[str(name)]

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
