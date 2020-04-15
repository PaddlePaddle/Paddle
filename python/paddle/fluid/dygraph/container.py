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
    'Sequential', 'ParameterList', 'LayerList', 'ParameterDict', 'LayerDict'
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


class ParameterDict(Layer):
    """
    Holds parameters in a dictionary.

    ``ParameterDict`` can be indexed like a regular Python dictionary, but parameters it
    contains are properly registered, and will be visible by all Layer methods.

    ParameterDict is an **ordered** dictionary that respects

    * the order of insertion, and

    * in ``ParameterDict.update``, the order of the merged ``OrderedDict``
      or another ``ParameterDict`` (the argument to ``ParameterDict.update`).

    Note that ``ParameterDict.update`` with other unordered mapping
    types (e.g., Python's plain ``dict``) does not preserve the order of the
    merged mapping.

    Parameters:
        parameters (iterable, optional): a mapping (dictionary) of
            (string : Parameter) or an iterable of key-value pairs
            of type (string, Parameter)

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            class MyLayer(fluid.Layer):
                def __init__(self):
                    super(MyLayer, self).__init__()
                    self.parameter_dict = fluid.dygraph.ParameterDict({
                        'param1':
                        fluid.layers.create_parameter(shape=[5, 10], dtype='float32'),
                        'param2':
                        fluid.layers.create_parameter(shape=[5, 5], dtype='float32')
                    })

                def forward(self, x, key):
                    tmp = self._helper.create_variable_for_type_inference('float32')
                    self._helper.append_op(
                                        type="mul",
                                        inputs={"X": x,
                                                "Y": self.parameter_dict[key]},
                                        outputs={"Out": tmp},
                                        attrs={"x_num_col_dims": 1,
                                               "y_num_col_dims": 1})
                    x = tmp
                    return x

            data_np = np.random.uniform(-1, 1, [3, 5]).astype('float32')
            with fluid.dygraph.guard():
                x = fluid.dygraph.to_variable(data_np)
                model = MyLayer()
                model_param1 = model(x, 'param1')
                model_param2 = model(x, 'param2')
                # model_param1.shape is [3, 10]
                print(model_param1.shape)
                # model_param2.shape is [3, 5]
                print(model_param2.shape)
    """

    def __init__(self, parameters=None):
        super(ParameterDict, self).__init__()
        if parameters is not None:
            self._update(parameters)

    def __getitem__(self, key):
        return self._parameters[key]

    def __setitem__(self, key, parameter):
        self.add_parameter(key, parameter)

    def __delitem__(self, key):
        del self._parameters[key]

    def __len__(self):
        return len(self._parameters)

    def __iter__(self):
        return iter(self._parameters.keys())

    def __contains__(self, key):
        return key in self._parameters

    def clear(self):
        """
        Remove all items from the ParameterDict.

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                import numpy as np

                class MyLayer(fluid.Layer):
                    def __init__(self):
                        super(MyLayer, self).__init__()
                        self.parameter_dict = fluid.dygraph.ParameterDict({
                            'param1':
                            fluid.layers.create_parameter(shape=[5, 10], dtype='float32'),
                            'param2':
                            fluid.layers.create_parameter(shape=[5, 5], dtype='float32')
                        })

                    def forward(self, x, key):
                        tmp = self._helper.create_variable_for_type_inference('float32')
                        self._helper.append_op(
                                            type="mul",
                                            inputs={"X": x,
                                                    "Y": self.parameter_dict[key]},
                                            outputs={"Out": tmp},
                                            attrs={"x_num_col_dims": 1,
                                                   "y_num_col_dims": 1})
                        x = tmp
                        return x

                with fluid.dygraph.guard():
                    model = MyLayer()
                    # len(model.parameter_dict) is 2
                    print(len(model.parameter_dict) == 2)
                    model.parameter_dict.clear()
                    # len(model.parameter_dict) is 0
                    print(len(model.parameter_dict) == 0)
        """
        self._parameters.clear()

    def pop(self, key):
        """
        Remove key from the ParameterDict and return its parameter.

        Parameters:
            key (string): key to pop from the ParameterDict

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                import numpy as np

                class MyLayer(fluid.Layer):
                    def __init__(self):
                        super(MyLayer, self).__init__()
                        self.parameter_dict = fluid.dygraph.ParameterDict({
                            'param1':
                            fluid.layers.create_parameter(shape=[5, 10], dtype='float32'),
                            'param2':
                            fluid.layers.create_parameter(shape=[5, 5], dtype='float32')
                        })

                    def forward(self, x, key):
                        tmp = self._helper.create_variable_for_type_inference('float32')
                        self._helper.append_op(
                                            type="mul",
                                            inputs={"X": x,
                                                    "Y": self.parameter_dict[key]},
                                            outputs={"Out": tmp},
                                            attrs={"x_num_col_dims": 1,
                                                   "y_num_col_dims": 1})
                        x = tmp
                        return x

                with fluid.dygraph.guard():
                    model = MyLayer()
                    # len(model.parameter_dict) is 2
                    print(len(model.parameter_dict) == 2)
                    model.parameter_dict.pop('param1')
                    # len(model.parameter_dict) is 1
                    print(len(model.parameter_dict) == 1)
        """
        value = self[key]
        del self[key]
        return value

    def keys(self):
        """
        Return an iterable of the ParameterDict keys.

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                import numpy as np

                class MyLayer(fluid.Layer):
                    def __init__(self):
                        super(MyLayer, self).__init__()
                        self.parameter_dict = fluid.dygraph.ParameterDict({
                            'param1':
                            fluid.layers.create_parameter(shape=[5, 10], dtype='float32'),
                            'param2':
                            fluid.layers.create_parameter(shape=[5, 5], dtype='float32')
                        })

                    def forward(self, x, key):
                        tmp = self._helper.create_variable_for_type_inference('float32')
                        self._helper.append_op(
                                            type="mul",
                                            inputs={"X": x,
                                                    "Y": self.parameter_dict[key]},
                                            outputs={"Out": tmp},
                                            attrs={"x_num_col_dims": 1,
                                                   "y_num_col_dims": 1})
                        x = tmp
                        return x

                with fluid.dygraph.guard():
                    model = MyLayer()
                    # param1
                    print(model.parameter_dict.keys()[0])
                    # param2
                    print(model.parameter_dict.keys()[1])
        """
        return self._parameters.keys()

    def items(self):
        """
        Return an iterable of the ParameterDict key/value pairs.

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                import numpy as np

                class MyLayer(fluid.Layer):
                    def __init__(self):
                        super(MyLayer, self).__init__()
                        self.parameter_dict = fluid.dygraph.ParameterDict({
                            'param1':
                            fluid.layers.create_parameter(shape=[5, 10], dtype='float32'),
                            'param2':
                            fluid.layers.create_parameter(shape=[5, 5], dtype='float32')
                        })

                    def forward(self, x, key):
                        tmp = self._helper.create_variable_for_type_inference('float32')
                        self._helper.append_op(
                                            type="mul",
                                            inputs={"X": x,
                                                    "Y": self.parameter_dict[key]},
                                            outputs={"Out": tmp},
                                            attrs={"x_num_col_dims": 1,
                                                   "y_num_col_dims": 1})
                        x = tmp
                        return x

                with fluid.dygraph.guard():
                    model = MyLayer()
                    # ('param1', parameter object)
                    print(model.parameter_dict.items()[0])
                    # ('param2', parameter object)
                    print(model.parameter_dict.items()[1])
        """
        return self._parameters.items()

    def values(self):
        """
        Return an iterable of the ParameterDict values.

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                import numpy as np

                class MyLayer(fluid.Layer):
                    def __init__(self):
                        super(MyLayer, self).__init__()
                        self.parameter_dict = fluid.dygraph.ParameterDict({
                            'param1':
                            fluid.layers.create_parameter(shape=[5, 10], dtype='float32'),
                            'param2':
                            fluid.layers.create_parameter(shape=[5, 5], dtype='float32')
                        })

                    def forward(self, x, key):
                        tmp = self._helper.create_variable_for_type_inference('float32')
                        self._helper.append_op(
                                            type="mul",
                                            inputs={"X": x,
                                                    "Y": self.parameter_dict[key]},
                                            outputs={"Out": tmp},
                                            attrs={"x_num_col_dims": 1,
                                                   "y_num_col_dims": 1})
                        x = tmp
                        return x

                with fluid.dygraph.guard():
                    model = MyLayer()
                    # parameter object
                    print(model.parameter_dict.values()[0])
                    # parameter object
                    print(model.parameter_dict.values()[1])
        """
        return self._parameters.values()

    def _update(self, parameters):
        """
        Update the ParameterDict with the key-value pairs from a
        mapping or an iterable, overwriting existing keys.

        .. note::
            If :attr:`parameters` is an ``OrderedDict``, a ParameterDict, or
            an iterable of key-value pairs, the order of new elements in it is preserved.

        Parameters:
            parameters (iterable): a dictionary from string to Parameter, or an iterable of
                key-value pairs of type (string, Parameter)
        """
        assert isinstance(
            parameters, (OrderedDict, ParameterDict, dict, list, tuple)
        ), "'parameters' argument must be type of OrderedDict, ParameterDict, dict, list, or tuple"
        if isinstance(parameters, (OrderedDict, ParameterDict)):
            for k, p in parameters.items():
                self.add_parameter(k, p)
        elif isinstance(parameters, dict):
            for k, p in sorted(parameters.items()):
                self.add_parameter(k, p)
        else:
            for p in parameters:
                assert len(
                    p
                ) == 2, "ParameterDict update with list or tuple requires elements with length 2"
                self.add_parameter(p[0], p[1])


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


class LayerDict(Layer):
    """
    LayerDict holds sublayers, and sublayers it contains are properly registered.
    Holded sublayers can be indexed like a regular Python dictionary.

    LayerDict is an **ordered** dictionary that respects

    * the order of insertion, and

    * in ``LayerDict.update``, the order of the merged ``OrderedDict``
      or another ``LayerDict`` (the argument to ``LayerDict.update``).

    Note that ``LayerDict.update`` with other unordered mapping
    types (e.g., Python's plain ``dict``) does not preserve the order of the
    merged mapping.

    Parameters:
        sublayers (iterable of Layer, optional): sublayers to hold

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            class MyLayer(fluid.Layer):
                def __init__(self):
                    super(MyLayer, self).__init__()
                    self.layerdict = fluid.dygraph.LayerDict({
                        'linear1':
                        fluid.dygraph.Linear(5, 10),
                        'linear2':
                        fluid.dygraph.Linear(5, 15)
                    })

                def forward(self, x, key):
                    return self.layerdict[key](x)

            data_np = np.random.uniform(-1, 1, [3, 5]).astype('float32')
            with fluid.dygraph.guard():
                x = fluid.dygraph.to_variable(data_np)
                model = MyLayer()
                model_linear1 = model(x, 'linear1')
                model_linear2 = model(x, 'linear2')
                # model_linear1.shape is [3, 10]
                print(model_linear1.shape)
                # model_linear2.shape is [3, 15]
                print(model_linear2.shape)
    """

    def __init__(self, layers=None):
        super(LayerDict, self).__init__()
        if layers is not None:
            self._update(layers)

    def __getitem__(self, key):
        return self._sub_layers[key]

    def __setitem__(self, key, layer):
        self.add_sublayer(key, layer)

    def __delitem__(self, key):
        del self._sub_layers[key]

    def __len__(self):
        return len(self._sub_layers)

    def __iter__(self):
        return iter(self._sub_layers)

    def __contains__(self, key):
        return key in self._sub_layers

    def clear(self):
        """
        Remove all items from the LayerDict.
    
        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                import numpy as np

                class MyLayer(fluid.Layer):
                    def __init__(self):
                        super(MyLayer, self).__init__()
                        self.layerdict = fluid.dygraph.LayerDict({
                            'linear1':
                            fluid.dygraph.Linear(5, 10),
                            'linear2':
                            fluid.dygraph.Linear(5, 15)
                        })

                    def forward(self, x, key):
                        return self.layerdict[key](x)

                with fluid.dygraph.guard():
                    model = MyLayer()
                    # len(model.layerdict) is 2
                    print(len(model.layerdict) == 2)
                    model.layerdict.clear()
                    # len(model.layerdict) is 0
                    print(len(model.layerdict) == 0)
        """
        self._sub_layers.clear()

    def pop(self, key):
        """
        Remove key from the LayerDict and return its layer.

        Parameters:
            key (string): key to pop from the LayerDict
        
        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                import numpy as np

                class MyLayer(fluid.Layer):
                    def __init__(self):
                        super(MyLayer, self).__init__()
                        self.layerdict = fluid.dygraph.LayerDict({
                            'linear1':
                            fluid.dygraph.Linear(5, 10),
                            'linear2':
                            fluid.dygraph.Linear(5, 15)
                        })

                    def forward(self, x, key):
                        return self.layerdict[key](x)

                with fluid.dygraph.guard():
                    model = MyLayer()
                    # len(model.layerdict) is 2
                    print(len(model.layerdict) == 2)
                    model.layerdict.pop('linear1')
                    # len(model.layerdict) is 0
                    print(len(model.layerdict) == 1)
        """
        v = self[key]
        del self[key]
        return v

    def keys(self):
        """
        Return an iterable of the LayerDict keys.

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                import numpy as np

                class MyLayer(fluid.Layer):
                    def __init__(self):
                        super(MyLayer, self).__init__()
                        self.layerdict = fluid.dygraph.LayerDict({
                            'linear1':
                            fluid.dygraph.Linear(5, 10),
                            'linear2':
                            fluid.dygraph.Linear(5, 15)
                        })

                    def forward(self, x, key):
                        return self.layerdict[key](x)

                with fluid.dygraph.guard():
                    model = MyLayer()
                    # linear1
                    print(model.layerdict.keys()[0])
                    # linear2
                    print(model.layerdict.keys()[1])
        """
        return self._sub_layers.keys()

    def items(self):
        """
        Return an iterable of the LayerDict key/value pairs.

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                import numpy as np

                class MyLayer(fluid.Layer):
                    def __init__(self):
                        super(MyLayer, self).__init__()
                        self.layerdict = fluid.dygraph.LayerDict({
                            'linear1':
                            fluid.dygraph.Linear(5, 10),
                            'linear2':
                            fluid.dygraph.Linear(5, 15)
                        })

                    def forward(self, x, key):
                        return self.layerdict[key](x)

                with fluid.dygraph.guard():
                    model = MyLayer()
                    # ('linear1', <paddle.fluid.dygraph.nn.Linear object>)
                    print(model.layerdict.items()[0])
                    # ('linear2', <paddle.fluid.dygraph.nn.Linear object>)
                    print(model.layerdict.items()[1])
        """
        return self._sub_layers.items()

    def values(self):
        """
        Return an iterable of the LayerDict values.

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                import numpy as np

                class MyLayer(fluid.Layer):
                    def __init__(self):
                        super(MyLayer, self).__init__()
                        self.layerdict = fluid.dygraph.LayerDict({
                            'linear1':
                            fluid.dygraph.Linear(5, 10),
                            'linear2':
                            fluid.dygraph.Linear(5, 15)
                        })

                    def forward(self, x, key):
                        return self.layerdict[key](x)

                with fluid.dygraph.guard():
                    model = MyLayer()
                    # <paddle.fluid.dygraph.nn.Linear object>
                    print(model.layerdict.values()[0])
                    # <paddle.fluid.dygraph.nn.Linear object>
                    print(model.layerdict.values()[1])
        """
        return self._sub_layers.values()

    def _update(self, layers):
        """
        Update the LayerDict with the key-value pairs from a  or an OrderedDict or  iterable, overwriting existing keys.

        .. note::
            If :attr:`layers` is an ``OrderedDict``, a ``LayerDict``, or
            an iterable of key-value pairs, the order of new elements in it is preserved.

        Parameters:
            other (iterable): a mapping (dictionary) from string to Layer,
                or an iterable of key-value pairs of type (string, Layer)
        """
        assert isinstance(
            layers, (OrderedDict, LayerDict, dict, list, tuple)
        ), "'layers' argument must be type of OrderedDict, LayerDict, dict, list, or tuple"
        if isinstance(layers, (OrderedDict, LayerDict)):
            for k, l in layers.items():
                self.add_sublayer(k, l)
        elif isinstance(layers, dict):
            for k, l in sorted(layers.items()):
                self.add_sublayer(k, l)
        else:
            for l in layers:
                assert len(
                    l
                ) == 2, "ParameterDict update with list or tuple requires elements with length 2"
                self.add_sublayer(l[0], l[1])

    def forward(self):
        raise NotImplementedError()


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
