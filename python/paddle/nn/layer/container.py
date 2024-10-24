# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import typing
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

from typing_extensions import Self

from ...base.dygraph.base import param_guard
from ...base.framework import Parameter
from .layers import Layer

if typing.TYPE_CHECKING:
    from paddle import Tensor

__all__ = []


class LayerDict(Layer):
    """
    LayerDict holds sublayers in the ordered dictionary, and sublayers it contains are properly registered.
    Held sublayers can be accessed like a regular ordered python dictionary.

    Parameters:
        sublayers (LayerDict|OrderedDict|list[(key,Layer)...], optional): iterable of key/value pairs, the type of value is 'paddle.nn.Layer' .

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import numpy as np
            >>> from collections import OrderedDict

            >>> sublayers = OrderedDict([
            ...     ('conv1d', paddle.nn.Conv1D(3, 2, 3)),
            ...     ('conv2d', paddle.nn.Conv2D(3, 2, 3)),
            ...     ('conv3d', paddle.nn.Conv3D(4, 6, (3, 3, 3))),
            >>> ])

            >>> layers_dict = paddle.nn.LayerDict(sublayers=sublayers)

            >>> l = layers_dict['conv1d']

            >>> for k in layers_dict:
            ...     l = layers_dict[k]
            ...
            >>> print(len(layers_dict))
            3

            >>> del layers_dict['conv2d']
            >>> print(len(layers_dict))
            2

            >>> conv1d = layers_dict.pop('conv1d')
            >>> print(len(layers_dict))
            1

            >>> layers_dict.clear()
            >>> print(len(layers_dict))
            0

    """

    def __init__(
        self,
        sublayers: (
            LayerDict
            | typing.Mapping[str, Layer]
            | Sequence[tuple[str, Layer]]
            | None
        ) = None,
    ) -> None:
        super().__init__()
        if sublayers is not None:
            self.update(sublayers)

    def __getitem__(self, key: str) -> Layer:
        return self._sub_layers[key]

    def __setitem__(self, key: str, sublayer: Layer) -> Layer:
        return self.add_sublayer(key, sublayer)

    def __delitem__(self, key: str) -> None:
        del self._sub_layers[key]

    def __len__(self) -> int:
        return len(self._sub_layers)

    def __iter__(self) -> Iterator[str]:
        return iter(self._sub_layers)

    def __contains__(self, key: str) -> bool:
        return key in self._sub_layers

    def clear(self) -> None:
        """
        Clear all the sublayers in the LayerDict.

        Parameters:
            None.

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> from collections import OrderedDict

                >>> sublayers = OrderedDict([
                ...     ('conv1d', paddle.nn.Conv1D(3, 2, 3)),
                ...     ('conv2d', paddle.nn.Conv2D(3, 2, 3)),
                ...     ('conv3d', paddle.nn.Conv3D(4, 6, (3, 3, 3))),
                >>> ])

                >>> layer_dict = paddle.nn.LayerDict(sublayers=sublayers)
                >>> len(layer_dict)
                3

                >>> layer_dict.clear()
                >>> len(layer_dict)
                0

        """
        self._sub_layers.clear()

    def pop(self, key: str) -> Layer:
        """
        Remove the key from the LayerDict and return the layer of the key.

        Parameters:
            key (str): the key to be removed.

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> from collections import OrderedDict

                >>> sublayers = OrderedDict([
                ...     ('conv1d', paddle.nn.Conv1D(3, 2, 3)),
                ...     ('conv2d', paddle.nn.Conv2D(3, 2, 3)),
                ...     ('conv3d', paddle.nn.Conv3D(4, 6, (3, 3, 3))),
                >>> ])

                >>> layer_dict = paddle.nn.LayerDict(sublayers=sublayers)
                >>> len(layer_dict)
                3

                >>> layer_dict.pop('conv2d')
                >>> len(layer_dict)
                2

        """
        v = self[key]
        del self[key]
        return v

    def keys(self) -> Iterable[str]:
        """
        Return the iterable of the keys in LayerDict.

        Parameters:
            None.

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> from collections import OrderedDict

                >>> sublayers = OrderedDict([
                ...     ('conv1d', paddle.nn.Conv1D(3, 2, 3)),
                ...     ('conv2d', paddle.nn.Conv2D(3, 2, 3)),
                ...     ('conv3d', paddle.nn.Conv3D(4, 6, (3, 3, 3))),
                >>> ])

                >>> layer_dict = paddle.nn.LayerDict(sublayers=sublayers)
                >>> for k in layer_dict.keys():
                ...     print(k)
                conv1d
                conv2d
                conv3d

        """
        return self._sub_layers.keys()

    def items(self) -> Iterable[tuple[str, Layer]]:
        """
        Return the iterable of the key/value pairs in LayerDict.

        Parameters:
            None.

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> from collections import OrderedDict

                >>> sublayers = OrderedDict([
                ...     ('conv1d', paddle.nn.Conv1D(3, 2, 3)),
                ...     ('conv2d', paddle.nn.Conv2D(3, 2, 3)),
                ...     ('conv3d', paddle.nn.Conv3D(4, 6, (3, 3, 3))),
                >>> ])

                >>> layer_dict = paddle.nn.LayerDict(sublayers=sublayers)
                >>> for k, v in layer_dict.items():
                ...     print(f"{k}:", v)
                conv1d : Conv1D(3, 2, kernel_size=[3], data_format=NCL)
                conv2d : Conv2D(3, 2, kernel_size=[3, 3], data_format=NCHW)
                conv3d : Conv3D(4, 6, kernel_size=[3, 3, 3], data_format=NCDHW)

        """
        return self._sub_layers.items()

    def values(self) -> Iterable[Layer]:
        """
        Return the iterable of the values in LayerDict.

        Parameters:
            None.

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> from collections import OrderedDict

                >>> sublayers = OrderedDict([
                ...     ('conv1d', paddle.nn.Conv1D(3, 2, 3)),
                ...     ('conv2d', paddle.nn.Conv2D(3, 2, 3)),
                ...     ('conv3d', paddle.nn.Conv3D(4, 6, (3, 3, 3))),
                >>> ])

                >>> layer_dict = paddle.nn.LayerDict(sublayers=sublayers)
                >>> for v in layer_dict.values():
                ...     print(v)
                Conv1D(3, 2, kernel_size=[3], data_format=NCL)
                Conv2D(3, 2, kernel_size=[3, 3], data_format=NCHW)
                Conv3D(4, 6, kernel_size=[3, 3, 3], data_format=NCDHW)

        """
        return self._sub_layers.values()

    def update(
        self,
        sublayers: (
            LayerDict | typing.Mapping[str, Layer] | Sequence[tuple[str, Layer]]
        ),
    ) -> None:
        """
        Update the key/values pairs in sublayers to the LayerDict, overwriting the existing keys.

        Parameters:
            sublayers (LayerDict|OrderedDict|list[(key,Layer)...]): iterable of key/value pairs, the type of value is 'paddle.nn.Layer' .

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> from collections import OrderedDict

                >>> sublayers = OrderedDict([
                ...     ('conv1d', paddle.nn.Conv1D(3, 2, 3)),
                ...     ('conv2d', paddle.nn.Conv2D(3, 2, 3)),
                ...     ('conv3d', paddle.nn.Conv3D(4, 6, (3, 3, 3))),
                >>> ])

                >>> new_sublayers = OrderedDict([
                ...     ('relu', paddle.nn.ReLU()),
                ...     ('conv2d', paddle.nn.Conv2D(4, 2, 4)),
                >>> ])
                >>> layer_dict = paddle.nn.LayerDict(sublayers=sublayers)

                >>> layer_dict.update(new_sublayers)

                >>> for k, v in layer_dict.items():
                ...     print(f"{k}:", v)
                conv1d : Conv1D(3, 2, kernel_size=[3], data_format=NCL)
                conv2d : Conv2D(4, 2, kernel_size=[4, 4], data_format=NCHW)
                conv3d : Conv3D(4, 6, kernel_size=[3, 3, 3], data_format=NCDHW)
                relu : ReLU()

        """

        assert isinstance(sublayers, Iterable), (
            "The type of sublayers is not iterable of key/value pairs, the type of sublayers is "
            + type(sublayers).__name__
        )

        if isinstance(sublayers, (OrderedDict, LayerDict, Mapping)):
            for key, layer in sublayers.items():
                self.add_sublayer(key, layer)
        else:
            # handle this format [(key1, layer1), (key2, layer2)...]
            for i, kv in enumerate(sublayers):
                if len(kv) != 2:
                    raise ValueError(
                        "The length of the "
                        + str(i)
                        + "'s element in sublayers is "
                        + str(len(kv))
                        + ", which must be 2."
                    )
                self.add_sublayer(kv[0], kv[1])


class ParameterDict(Layer):
    """
    Holds parameters in a dictionary.

    ParameterDict can be indexed like a regular Python dictionary, but Parameters it contains are properly registered.

    Parameters:
        parameters (iterable, optional): a mapping (dictionary) of (string : Any) or an iterable of key-value pairs of type (string, Any)

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> class MyLayer(paddle.nn.Layer):
            ...     def __init__(self, num_stacked_param):
            ...         super().__init__()
            ...         # create ParameterDict with iterable Parameters
            ...         self.params = paddle.nn.ParameterDict(
            ...             {f"t{i}": paddle.create_parameter(shape=[2, 2], dtype='float32') for i in range(num_stacked_param)})
            ...
            ...     def forward(self, x):
            ...         for i, key in enumerate(self.params):
            ...             x = paddle.matmul(x, self.params[key])
            ...         return x
            ...
            >>> x = paddle.uniform(shape=[5, 2], dtype='float32')
            >>> num_stacked_param = 4
            >>> model = MyLayer(num_stacked_param)
            >>> print(len(model.params))
            4
            >>> res = model(x)
            >>> print(res.shape)
            [5, 2]

            >>> replaced_param = paddle.create_parameter(shape=[2, 3], dtype='float32')
            >>> model.params['t3'] = replaced_param  # replace t3 param
            >>> res = model(x)
            >>> print(res.shape)
            [5, 3]
            >>> model.params['t4'] = paddle.create_parameter(shape=[3, 4], dtype='float32')  # append param
            >>> print(len(model.params))
            5
            >>> res = model(x)
            >>> print(res.shape)
            [5, 4]
    """

    def __init__(
        self,
        parameters: (
            ParameterDict
            | Mapping[str, Tensor]
            | Sequence[tuple[str, Tensor]]
            | None
        ) = None,
    ) -> None:
        super().__init__()
        if parameters is not None:
            self.update(parameters)

    def __getitem__(self, key: str) -> Tensor:
        with param_guard(self._parameters):
            return self._parameters[key]

    def __setitem__(self, key: str, param: Tensor) -> None:
        assert isinstance(param, Parameter)
        setattr(self, key, param)

    def __len__(self) -> int:
        return len(self._parameters)

    def __iter__(self) -> Iterator[str]:
        return iter(self._parameters)

    def update(
        self,
        parameters: (
            ParameterDict | Mapping[str, Tensor] | Sequence[tuple[str, Tensor]]
        ),
    ) -> None:
        """Update a given parameter at the end of the dict.

        Parameters:
            parameters (Parameter): parameter to update
        """
        assert isinstance(parameters, Iterable), (
            "The type of parameters is not iterable of key/value pairs, the type of sublayers is "
            + type(parameters).__name__
        )

        if isinstance(parameters, (OrderedDict, ParameterDict, Mapping)):
            for key, parameter in parameters.items():
                self.add_parameter(key, parameter)
        else:
            for i, kv in enumerate(parameters):
                if len(kv) != 2:
                    raise ValueError(
                        f"The length of the {i}'s element in parameters is {len(kv)}, which must be 2."
                    )
                self.add_parameter(kv[0], kv[1])


class ParameterList(Layer):
    """ParameterList Container.

    This container acts like a Python list, but parameters it contains will be properly added.

    Parameters:
        parameters (iterable, optional): Iterable Parameters to be added.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> class MyLayer(paddle.nn.Layer):
            ...     def __init__(self, num_stacked_param):
            ...         super().__init__()
            ...         # create ParameterList with iterable Parameters
            ...         self.params = paddle.nn.ParameterList(
            ...             [paddle.create_parameter(
            ...                 shape=[2, 2], dtype='float32')] * num_stacked_param)
            ...
            ...     def forward(self, x):
            ...         for i, p in enumerate(self.params):
            ...             tmp = self._helper.create_variable_for_type_inference('float32')
            ...             self._helper.append_op(
            ...                 type="mul",
            ...                 inputs={"X": x,
            ...                         "Y": p},
            ...                 outputs={"Out": tmp},
            ...                 attrs={"x_num_col_dims": 1,
            ...                         "y_num_col_dims": 1})
            ...             x = tmp
            ...         return x
            ...
            >>> x = paddle.uniform(shape=[5, 2], dtype='float32')
            >>> num_stacked_param = 4
            >>> model = MyLayer(num_stacked_param)
            >>> print(len(model.params))
            4
            >>> res = model(x)
            >>> print(res.shape)
            [5, 2]

            >>> replaced_param = paddle.create_parameter(shape=[2, 3], dtype='float32')
            >>> model.params[num_stacked_param - 1] = replaced_param  # replace last param
            >>> res = model(x)
            >>> print(res.shape)
            [5, 3]
            >>> model.params.append(paddle.create_parameter(shape=[3, 4], dtype='float32'))  # append param
            >>> print(len(model.params))
            5
            >>> res = model(x)
            >>> print(res.shape)
            [5, 4]
    """

    def __init__(self, parameters: Iterable[Tensor] | None = None) -> None:
        super().__init__()
        if parameters is not None:
            for idx, param in enumerate(parameters):
                assert isinstance(param, Parameter)
                self.add_parameter(str(idx), param)

    def __getitem__(self, idx: int) -> Tensor:
        with param_guard(self._parameters):
            return self._parameters[str(idx)]

    def __setitem__(self, idx: int, param: Tensor) -> None:
        assert isinstance(param, Parameter)
        setattr(self, str(idx), param)

    def __len__(self) -> int:
        return len(self._parameters)

    def __iter__(self) -> Iterator[Tensor]:
        with param_guard(self._parameters):
            return iter(self._parameters.values())

    def append(self, parameter: Tensor) -> Self:
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

            >>> import paddle

            >>> class MyLayer(paddle.nn.Layer):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.linears = paddle.nn.LayerList(
            ...             [paddle.nn.Linear(10, 10) for i in range(10)])
            ...
            ...     def forward(self, x):
            ...         # LayerList can act as an iterable, or be indexed using ints
            ...         for i, l in enumerate(self.linears):
            ...             x = self.linears[i // 2](x) + l(x)
            ...         return x
    """

    def __init__(self, sublayers: Iterable[Layer] | None = None) -> None:
        super().__init__()
        if sublayers is not None:
            for idx, layer in enumerate(sublayers):
                self.add_sublayer(str(idx), layer)

    def _get_abs_idx(self, idx: int) -> int:
        if isinstance(idx, int):
            if not (-len(self) <= idx < len(self)):
                raise IndexError(
                    f'index {idx} is out of range, should be an integer in range [{-len(self)}, {len(self)})'
                )
            if idx < 0:
                idx += len(self)
        return idx

    def __getitem__(self, idx: int) -> Layer:
        if isinstance(idx, slice):
            return self.__class__(list(self._sub_layers.values())[idx])
        else:
            idx = self._get_abs_idx(idx)
            return self._sub_layers[str(idx)]

    def __setitem__(self, idx: int, sublayer: Layer) -> None:
        idx = self._get_abs_idx(idx)
        return setattr(self, str(idx), sublayer)

    def __delitem__(self, idx: int) -> None:
        if isinstance(idx, slice):
            for k in range(len(self._sub_layers))[idx]:
                delattr(self, str(k))
        else:
            idx = self._get_abs_idx(idx)
            delattr(self, str(idx))
        str_indices = [str(i) for i in range(len(self._sub_layers))]
        self._sub_layers = OrderedDict(
            list(zip(str_indices, self._sub_layers.values()))
        )

    def __len__(self) -> int:
        return len(self._sub_layers)

    def __iter__(self) -> Iterator[Layer]:
        return iter(self._sub_layers.values())

    def append(self, sublayer: Layer) -> Self:
        """
        Appends a sublayer to the end of the list.

        Parameters:
            sublayer (Layer): sublayer to append

        Examples:
            .. code-block:: python

                >>> import paddle

                >>> linears = paddle.nn.LayerList([paddle.nn.Linear(10, 10) for i in range(10)])
                >>> another = paddle.nn.Linear(10, 10)
                >>> linears.append(another)
                >>> print(len(linears))
                11
        """
        self.add_sublayer(str(len(self)), sublayer)
        return self

    def insert(self, index: int, sublayer: Layer) -> None:
        """
        Insert a sublayer before a given index in the list.

        Parameters:
            index (int): index to insert.
            sublayer (Layer): sublayer to insert

        Examples:
            .. code-block:: python

                >>> import paddle

                >>> linears = paddle.nn.LayerList([paddle.nn.Linear(10, 10) for i in range(10)])
                >>> another = paddle.nn.Linear(10, 10)
                >>> linears.insert(3, another)
                >>> print(linears[3] is another)
                True
                >>> another = paddle.nn.Linear(10, 10)
                >>> linears.insert(-1, another)
                >>> print(linears[-2] is another)
                True
        """
        assert isinstance(index, int) and -len(self._sub_layers) <= index < len(
            self._sub_layers
        ), f"index should be an integer in range [{-len(self)}, {len(self)})"

        index = self._get_abs_idx(index)
        for i in range(len(self._sub_layers), index, -1):
            self._sub_layers[str(i)] = self._sub_layers[str(i - 1)]
        self._sub_layers[str(index)] = sublayer

    def extend(self, sublayers: Iterable[Layer]) -> Self:
        """
        Appends sublayers to the end of the list.

        Parameters:
            sublayers (iterable of Layer): iterable of sublayers to append

        Returns:
            None

        Examples:
            .. code-block:: python

                >>> import paddle

                >>> linears = paddle.nn.LayerList([paddle.nn.Linear(10, 10) for i in range(10)])
                >>> another_list = paddle.nn.LayerList([paddle.nn.Linear(10, 10) for i in range(5)])
                >>> linears.extend(another_list)
                >>> print(len(linears))
                15
                >>> print(another_list[0] is linears[10])
                True
        """
        offset = len(self)
        for i, sublayer in enumerate(sublayers):
            idx = str(offset + i)
            self.add_sublayer(idx, sublayer)
        return self


class Sequential(Layer):
    """Sequential container.
    Sub layers will be added to this container in the order of argument in the constructor.
    The argument passed to the constructor can be iterable Layers or iterable name Layer pairs.

    Parameters:
        layers(Layer|list|tuple): Layer or list/tuple of iterable name Layer pair.

    Returns:
        None.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> data = paddle.uniform(shape=[30, 10], dtype='float32')
            >>> # create Sequential with iterable Layers
            >>> model1 = paddle.nn.Sequential(
            ...     paddle.nn.Linear(10, 1), paddle.nn.Linear(1, 2)
            >>> )
            >>> model1[0]  # access the first layer
            >>> res1 = model1(data)  # sequential execution

            >>> # create Sequential with name Layer pairs
            >>> model2 = paddle.nn.Sequential(
            ...     ('l1', paddle.nn.Linear(10, 2)),
            ...     ('l2', paddle.nn.Linear(2, 3))
            >>> )
            >>> model2['l1']  # access l1 layer
            >>> model2.add_sublayer('l3', paddle.nn.Linear(3, 3))  # add sublayer
            >>> res2 = model2(data)  # sequential execution

            >>> # append single layer at the end of sequential
            >>> model2 = paddle.nn.Sequential(paddle.nn.Linear(10, 20))
            >>> model2.append(paddle.nn.Linear(20, 30))
            >>> res2 = model2(data)  # [30, 30]

            >>> # insert single layer at the given position
            >>> model2 = paddle.nn.Sequential(paddle.nn.Linear(20, 30))
            >>> model2.insert(0, paddle.nn.Linear(10, 20))
            >>> res2 = model2(data)  # [30, 30]

            >>> # extend sequential with given sequence of layer(s) at the end
            >>> model2 = paddle.nn.Sequential()
            >>> model2.extend([paddle.nn.Linear(10, 20), paddle.nn.Linear(20, 30)])
            >>> res2 = model2(data)  # [30, 30]
    """

    def __init__(self, *layers: Layer | tuple[str, Layer] | list[Any]) -> None:
        super().__init__()
        if len(layers) > 0 and isinstance(layers[0], (list, tuple)):
            for name, layer in layers:
                self.add_sublayer(name, layer)
        else:
            for idx, layer in enumerate(layers):
                self.add_sublayer(str(idx), layer)

    def __getitem__(self, name: str | slice | int) -> Layer:
        if isinstance(name, slice):
            return self.__class__(*(list(self._sub_layers.values())[name]))
        elif isinstance(name, str):
            return self._sub_layers[name]
        else:
            if name >= len(self._sub_layers):
                raise IndexError(f'index {name} is out of range')
            elif name < 0 and name >= -len(self._sub_layers):
                name += len(self._sub_layers)
            elif name < -len(self._sub_layers):
                raise IndexError(f'index {name} is out of range')
            return list(self._sub_layers.values())[name]

    def __setitem__(self, name: str, layer: Layer) -> None:
        assert isinstance(layer, Layer)
        setattr(self, str(name), layer)

    def __delitem__(self, name: str) -> None:
        name = str(name)
        assert name in self._sub_layers
        del self._sub_layers[name]

    def __len__(self) -> int:
        return len(self._sub_layers)

    def forward(self, input: Any) -> Any:
        for layer in self._sub_layers.values():
            input = layer(input)
        return input

    def append(self, module: Layer) -> Sequential:
        self.add_sublayer(str(len(self)), module)
        return self

    def insert(self, index: int, module: Layer) -> Sequential:
        if not isinstance(module, Layer):
            raise AssertionError(f'module should be of type: {Layer}')
        n = len(self._sub_layers)
        if not (-n <= index <= n):
            raise IndexError(f'Index out of range: {index}')
        if index < 0:
            index += n
        for i in range(n, index, -1):
            self._sub_layers[str(i)] = self._sub_layers[str(i - 1)]
        self._sub_layers[str(index)] = module
        return self

    def extend(self, sequential: Iterable[Layer]) -> Sequential:
        for layer in sequential:
            self.append(layer)
        return self

    def __iter__(self) -> Iterator[Layer]:
        return iter(self._sub_layers.values())
