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

from collections import OrderedDict
from ..framework import Parameter
from .layers import Layer

__all__ = [
    'Sequential',
    'ParameterList',
    'LayerList',
]


class Sequential(Layer):
    """Sequential container.
    Sub layers will be added to this container in the order of argument in the constructor.
    The argument passed to the constructor can be iterable Layers or iterable name Layer pairs.

    Parameters:
        *layers(tuple): Layers or iterable name Layer pairs.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            data = np.random.uniform(-1, 1, [30, 10]).astype('float32')
            with fluid.dygraph.guard():
                data = fluid.dygraph.to_variable(data)
                # create Sequential with iterable Layers
                model1 = fluid.dygraph.Sequential(
                    fluid.Linear(10, 1), fluid.Linear(1, 2)
                )
                model1[0]  # access the first layer
                res1 = model1(data)  # sequential execution

                # create Sequential with name Layer pairs
                model2 = fluid.dygraph.Sequential(
                    ('l1', fluid.Linear(10, 2)),
                    ('l2', fluid.Linear(2, 3))
                )
                model2['l1']  # access l1 layer
                model2.add_sublayer('l3', fluid.Linear(3, 3))  # add sublayer
                res2 = model2(data)  # sequential execution

    """

    def __init__(self, *layers):
        super(Sequential, self).__init__()
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


class ParameterList(Layer):
    """ParameterList Container.

    This container acts like a Python list, but parameters it contains will be properly added.

    Parameters:
        parameters (iterable, optional): Iterable Parameters to be added

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            class MyLayer(fluid.Layer):
                def __init__(self, num_stacked_param):
                    super(MyLayer, self).__init__()
                    # create ParameterList with iterable Parameters
                    self.params = fluid.dygraph.ParameterList(
                        [fluid.layers.create_parameter(
                            shape=[2, 2], dtype='float32')] * num_stacked_param)

                def forward(self, x):
                    for i, p in enumerate(self.params):
                        tmp = self._helper.create_variable_for_type_inference('float32')
                        self._helper.append_op(
                            type="mul",
                            inputs={"X": x,
                                    "Y": p},
                            outputs={"Out": tmp},
                            attrs={"x_num_col_dims": 1,
                                   "y_num_col_dims": 1})
                        x = tmp
                    return x

            data_np = np.random.uniform(-1, 1, [5, 2]).astype('float32')
            with fluid.dygraph.guard():
                x = fluid.dygraph.to_variable(data_np)
                num_stacked_param = 4
                model = MyLayer(num_stacked_param)
                print(len(model.params))  # 4
                res = model(x)
                print(res.shape)  # [5, 2]

                replaced_param = fluid.layers.create_parameter(shape=[2, 3], dtype='float32')
                model.params[num_stacked_param - 1] = replaced_param  # replace last param
                res = model(x)
                print(res.shape)  # [5, 3]
                model.params.append(fluid.layers.create_parameter(shape=[3, 4], dtype='float32'))  # append param
                print(len(model.params))  # 5
                res = model(x)
                print(res.shape)  # [5, 4]
    """

    def __init__(self, parameters=None):
        super(ParameterList, self).__init__()
        if parameters is not None:
            for idx, param in enumerate(parameters):
                assert isinstance(param, Parameter)
                self.add_parameter(str(idx), param)

    def __getitem__(self, idx):
        return self._parameters[str(idx)]

    def __setitem__(self, idx, param):
        assert isinstance(param, Parameter)
        setattr(self, str(idx), param)

    def __len__(self):
        return len(self._parameters)

    def __iter__(self):
        return iter(self._parameters.values())

    def append(self, parameter):
        """Appends a given parameter at the end of the list.

        Parameters:
            parameter (Parameter): parameter to append
        """
        idx = len(self._parameters)
        self.add_parameter(str(idx), parameter)
        return self


class LayerList(Layer):
    """
    LayerList holds sublayers, and sublayers it contains are properly registered.
    Holded sublayers can be indexed like a regular python list.

    Parameters:
        sublayers (iterable of Layer, optional): sublayers to hold

    Examples:
        .. code-block:: python
            import paddle.fluid as fluid
            import numpy as np

            class MyLayer(fluid.Layer):
                def __init__(self):
                    super(MyLayer, self).__init__()
                    self.linears = fluid.dygraph.LayerList(
                        [fluid.dygraph.Linear(10, 10) for i in range(10)])

                def forward(self, x):
                    # LayerList can act as an iterable, or be indexed using ints
                    for i, l in enumerate(self.linears):
                        x = self.linears[i // 2](x) + l(x)
                    return x
    """

    def __init__(self, sublayers=None):
        super(LayerList, self).__init__()
        if sublayers is not None:
            for idx, layer in enumerate(sublayers):
                self.add_sublayer(str(idx), layer)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(list(self._sub_layers.values())[idx])
        else:
            return self._sub_layers[str(idx)]

    def __setitem__(self, idx, sublayer):
        return setattr(self, str(idx), sublayer)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for k in range(len(self._sub_layers))[idx]:
                delattr(self, str(k))
        else:
            delattr(self, str(idx))
        str_indices = [str(i) for i in range(len(self._sub_layers))]
        self._sub_layers = OrderedDict(
            list(zip(str_indices, self._sub_layers.values())))

    def __len__(self):
        return len(self._sub_layers)

    def __iter__(self):
        return iter(self._sub_layers.values())

    def append(self, sublayer):
        """
        Appends a sublayer to the end of the list.

        Parameters:
            sublayer (Layer): sublayer to append

        Examples:
            .. code-block:: python
                import paddle.fluid as fluid

                with fluid.dygraph.guard():
                    linears = fluid.dygraph.LayerList([fluid.dygraph.Linear(10, 10) for i in range(10)])
                    another = fluid.dygraph.Linear(10, 10)
                    linears.append(another)
                    print(len(linears))  # 11
        """
        self.add_sublayer(str(len(self)), sublayer)
        return self

    def insert(self, index, sublayer):
        """
        Insert a sublayer before a given index in the list.

        Parameters:
            index (int): index to insert.
            sublayer (Layer): sublayer to insert

        Examples:
            .. code-block:: python
                import paddle.fluid as fluid

                with fluid.dygraph.guard():
                    linears = fluid.dygraph.LayerList([fluid.dygraph.Linear(10, 10) for i in range(10)])
                    another = fluid.dygraph.Linear(10, 10)
                    linears.insert(3, another)
                    print(linears[3] is another)  # True
        """
        assert isinstance(index, int) and \
               0 <= index < len(self._sub_layers), \
            "index should be an integer in range [0, len(self))"
        for i in range(len(self._sub_layers), index, -1):
            self._sub_layers[str(i)] = self._sub_layers[str(i - 1)]
        self._sub_layers[str(index)] = sublayer

    def extend(self, sublayers):
        """
        Appends sublayers to the end of the list.

        Parameters:
            sublayers (iterable of Layer): iterable of sublayers to append

        Examples:
            .. code-block:: python
                import paddle.fluid as fluid

                with fluid.dygraph.guard():
                    linears = fluid.dygraph.LayerList([fluid.dygraph.Linear(10, 10) for i in range(10)])
                    another_list = fluid.dygraph.LayerList([fluid.dygraph.Linear(10, 10) for i in range(5)])
                    linears.extend(another_list)
                    print(len(linears))  # 15
                    print(another_list[0] is linears[10])  # True
        """
        offset = len(self)
        for i, sublayer in enumerate(sublayers):
            idx = str(offset + i)
            self.add_sublayer(idx, sublayer)
        return self
