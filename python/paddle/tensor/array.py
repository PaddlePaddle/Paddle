#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

# Define functions about array.
from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, overload

import paddle
from paddle import _typing

from ..base.data_feeder import check_type, check_variable_and_dtype
from ..base.framework import in_pir_mode
from ..common_ops_import import Variable
from ..framework import LayerHelper, core, in_dynamic_mode

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = []
T = TypeVar("T")


@overload
def array_length(array: list[Any]) -> int: ...


@overload
def array_length(array: paddle.Tensor) -> paddle.Tensor: ...


def array_length(array):
    """
    This OP is used to get the length of the input array.

    Args:
        array (list|Tensor): The input array that will be used to compute the length. In dynamic mode, ``array`` is a Python list. But in static graph mode, array is a Tensor whose VarType is DENSE_TENSOR_ARRAY.

    Returns:
        Tensor, 0-D Tensor with shape [], which is the length of array.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> arr = paddle.tensor.create_array(dtype='float32')
            >>> x = paddle.full(shape=[3, 3], fill_value=5, dtype="float32")
            >>> i = paddle.zeros(shape=[1], dtype="int32")

            >>> arr = paddle.tensor.array_write(x, i, array=arr)

            >>> arr_len = paddle.tensor.array_length(arr)
            >>> print(arr_len)
            1
    """
    if in_dynamic_mode():
        assert isinstance(
            array, list
        ), "The 'array' in array_write must be a list in dygraph mode"
        return len(array)
    elif in_pir_mode():
        if (
            not isinstance(array, paddle.pir.Value)
            or not array.is_dense_tensor_array_type()
        ):
            raise TypeError(
                "array should be tensor array variable in array_length Op"
            )
        return paddle._pir_ops.array_length(array)
    else:
        if (
            not isinstance(array, Variable)
            or array.type != core.VarDesc.VarType.DENSE_TENSOR_ARRAY
        ):
            raise TypeError(
                "array should be tensor array variable in array_length Op"
            )

        helper = LayerHelper('array_length', **locals())
        tmp = helper.create_variable_for_type_inference(dtype='int64')
        tmp.stop_gradient = True
        helper.append_op(
            type='lod_array_length',
            inputs={'X': [array]},
            outputs={'Out': [tmp]},
        )
        return tmp


@overload
def array_read(array: list[T], i: paddle.Tensor) -> T: ...


@overload
def array_read(array: paddle.Tensor, i: paddle.Tensor) -> paddle.Tensor: ...


def array_read(array, i):
    """
    This OP is used to read data at the specified position from the input array.

    Case:

    .. code-block:: text

        Input:
            The shape of first three tensors are [1], and that of the last one is [1,2]:
                array = ([0.6], [0.1], [0.3], [0.4, 0.2])
            And:
                i = [3]

        Output:
            output = [0.4, 0.2]

    Args:
        array (list|Tensor): The input array. In dynamic mode, ``array`` is a Python list. But in static graph mode, array is a Tensor whose ``VarType`` is ``DENSE_TENSOR_ARRAY``.
        i (Tensor): 1-D Tensor, whose shape is [1] and dtype is int64. It represents the
            specified read position of ``array``.

    Returns:
        Tensor, A Tensor that is read at the specified position of ``array``.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> arr = paddle.tensor.create_array(dtype="float32")
            >>> x = paddle.full(shape=[1, 3], fill_value=5, dtype="float32")
            >>> i = paddle.zeros(shape=[1], dtype="int32")

            >>> arr = paddle.tensor.array_write(x, i, array=arr)

            >>> item = paddle.tensor.array_read(arr, i)
            >>> print(item.numpy())
            [[5. 5. 5.]]
    """
    if in_dynamic_mode():
        assert isinstance(
            array, list
        ), "The 'array' in array_read must be list in dygraph mode"
        assert isinstance(
            i, Variable
        ), "The index 'i' in array_read must be Variable in dygraph mode"
        assert i.shape == [
            1
        ], "The shape of index 'i' should be [1] in dygraph mode"
        i = i.item(0)
        return array[i]
    elif in_pir_mode():
        if (
            not isinstance(array, paddle.pir.Value)
            or not array.is_dense_tensor_array_type()
        ):
            raise TypeError(
                "array should be tensor array variable in array_length Op"
            )
        return paddle._pir_ops.array_read(array, i)
    else:
        check_variable_and_dtype(i, 'i', ['int64'], 'array_read')
        helper = LayerHelper('array_read', **locals())
        if (
            not isinstance(array, Variable)
            or array.type != core.VarDesc.VarType.DENSE_TENSOR_ARRAY
        ):
            raise TypeError("array should be tensor array variable")
        out = helper.create_variable_for_type_inference(dtype=array.dtype)
        helper.append_op(
            type='read_from_array',
            inputs={'X': [array], 'I': [i]},
            outputs={'Out': [out]},
        )
        return out


@overload
def array_write(
    x: paddle.Tensor, i: paddle.Tensor, array: None = None
) -> list[Any] | paddle.Tensor: ...


@overload
def array_write(
    x: paddle.Tensor, i: paddle.Tensor, array: list[paddle.Tensor]
) -> list[paddle.Tensor]: ...


@overload
def array_write(
    x: paddle.Tensor, i: paddle.Tensor, array: paddle.Tensor
) -> paddle.Tensor: ...


def array_write(
    x,
    i,
    array=None,
):
    """
    This OP writes the input ``x`` into the i-th position of the ``array`` returns the modified array.
    If ``array`` is none, a new array will be created and returned.

    Args:
        x (Tensor): The input data to be written into array. It's multi-dimensional
            Tensor. Data type: float32, float64, int32, int64 and bool.
        i (Tensor): 0-D Tensor with shape [], which represents the position into which
            ``x`` is written.
        array (list|Tensor, optional): The array into which ``x`` is written. The default value is None,
            when a new array will be created and returned as a result. In dynamic mode, ``array`` is a Python list.
            But in static graph mode, array is a Tensor whose ``VarType`` is ``DENSE_TENSOR_ARRAY``.

    Returns:
        list|Tensor, The input ``array`` after ``x`` is written into.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> arr = paddle.tensor.create_array(dtype="float32")
            >>> x = paddle.full(shape=[1, 3], fill_value=5, dtype="float32")
            >>> i = paddle.zeros(shape=[1], dtype="int32")

            >>> arr = paddle.tensor.array_write(x, i, array=arr)

            >>> item = paddle.tensor.array_read(arr, i)
            >>> print(item.numpy())
            [[5. 5. 5.]]
    """
    if in_dynamic_mode():
        assert isinstance(
            x, Variable
        ), "The input data 'x' in array_write must be Variable in dygraph mode"
        assert isinstance(
            i, Variable
        ), "The index 'i' in array_write must be Variable in dygraph mode"
        assert i.shape == [
            1
        ], "The shape of index 'i' should be [1] in dygraph mode"
        i = i.item(0)
        if array is None:
            array = create_array(x.dtype)
        assert isinstance(
            array, list
        ), "The 'array' in array_write must be a list in dygraph mode"
        assert i <= len(
            array
        ), "The index 'i' should not be greater than the length of 'array' in dygraph mode"
        if i < len(array):
            array[i] = x
        else:
            array.append(x)
        return array
    elif in_pir_mode():
        check_variable_and_dtype(i, 'i', ['int64'], 'array_write')
        if not isinstance(x, paddle.pir.Value):
            raise TypeError(f"x should be pir.Value, but received {type(x)}.")
        if array is not None:
            if (
                not isinstance(array, paddle.pir.Value)
                or not array.is_dense_tensor_array_type()
            ):
                raise TypeError("array should be tensor array variable")
        if array is None:
            array = paddle._pir_ops.create_array(x.dtype)

        if array.dtype != paddle.base.libpaddle.DataType.UNDEFINED:
            x = paddle.cast(x, array.dtype)
        paddle._pir_ops.array_write_(array, x, i)
        return array
    else:
        check_variable_and_dtype(i, 'i', ['int64'], 'array_write')
        check_type(x, 'x', (Variable), 'array_write')
        helper = LayerHelper('array_write', **locals())
        if array is not None:
            if (
                not isinstance(array, Variable)
                or array.type != core.VarDesc.VarType.DENSE_TENSOR_ARRAY
            ):
                raise TypeError(
                    "array should be tensor array variable in array_write Op"
                )
        if array is None:
            array = helper.create_variable(
                name=f"{helper.name}.out",
                type=core.VarDesc.VarType.DENSE_TENSOR_ARRAY,
                dtype=x.dtype,
            )
        helper.append_op(
            type='write_to_array',
            inputs={'X': [x], 'I': [i]},
            outputs={'Out': [array]},
        )
        return array


def create_array(
    dtype: _typing.DTypeLike,
    initialized_list: Sequence[paddle.Tensor] | None = None,
) -> paddle.Tensor | list[paddle.Tensor]:
    """
    This OP creates an array. It is used as the input of :ref:`api_paddle_tensor_array_array_read` and
    :ref:`api_paddle_tensor_array_array_write`.

    Args:
        dtype (str): The data type of the elements in the array. Support data type: float32, float64, int32, int64 and bool.
        initialized_list(list): Used to initialize as default value for created array.
                    All values in initialized list should be a Tensor.

    Returns:
        list|Tensor, An empty array. In dynamic mode, ``array`` is a Python list. But in static graph mode, array is a Tensor
        whose ``VarType`` is ``DENSE_TENSOR_ARRAY``.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> arr = paddle.tensor.create_array(dtype="float32")
            >>> x = paddle.full(shape=[1, 3], fill_value=5, dtype="float32")
            >>> i = paddle.zeros(shape=[1], dtype="int32")

            >>> arr = paddle.tensor.array_write(x, i, array=arr)

            >>> item = paddle.tensor.array_read(arr, i)
            >>> print(item.numpy())
            [[5. 5. 5.]]

    """
    array = []
    if initialized_list is not None:
        if not isinstance(initialized_list, (list, tuple)):
            raise TypeError(
                f"Require type(initialized_list) should be list/tuple, but received {type(initialized_list)}"
            )
        array = list(initialized_list)

    # NOTE: Only support plain list like [x, y,...], not support nested list in static graph mode.
    for val in array:
        if not isinstance(val, (Variable, paddle.pir.Value)):
            raise TypeError(
                f"All values in `initialized_list` should be Variable or pir.Value, but received {type(val)}."
            )

    if in_dynamic_mode():
        return array
    elif in_pir_mode():
        if not isinstance(dtype, (core.VarDesc.VarType, core.DataType)):
            dtype = paddle.base.framework.convert_np_dtype_to_dtype_(dtype)
        out = paddle._pir_ops.create_array(dtype)
        for val in array:
            if dtype != paddle.base.libpaddle.DataType.UNDEFINED:
                val = paddle.cast(val, dtype)
            paddle._pir_ops.array_write_(out, val, array_length(out))
        return out
    else:
        helper = LayerHelper("array", **locals())
        tensor_array: paddle.Tensor = helper.create_variable(
            name=f"{helper.name}.out",
            type=core.VarDesc.VarType.DENSE_TENSOR_ARRAY,
            dtype=dtype,
        )

        for val in array:
            array_write(x=val, i=array_length(tensor_array), array=tensor_array)

        return tensor_array
