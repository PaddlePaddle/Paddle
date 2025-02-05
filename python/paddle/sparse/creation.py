#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from typing import TYPE_CHECKING, Any

import numpy as np

import paddle
from paddle import _C_ops, in_dynamic_mode
from paddle.base.data_feeder import convert_dtype
from paddle.base.framework import (
    _current_expected_place,
    _get_paddle_place,
    core,
    dygraph_only,
    in_pir_mode,
)
from paddle.base.layer_helper import LayerHelper
from paddle.tensor import max, to_tensor

if TYPE_CHECKING:
    import numpy.typing as npt

    from paddle import CPUPlace, CUDAPinnedPlace, CUDAPlace, Tensor
    from paddle._typing import DTypeLike, NumericSequence, ShapeLike

__all__ = [
    'sparse_coo_tensor',
    'sparse_csr_tensor',
]


def _handle_dtype(data, dtype):
    if dtype:
        if convert_dtype(dtype) != convert_dtype(data.dtype):
            return data.astype(convert_dtype(dtype))
    return data


def _infer_dense_shape(indices, values):
    assert len(indices.shape) == 2
    lens = max(indices, axis=1)
    lens = lens + 1
    lens = lens.numpy()
    if len(values.shape) > 1:
        lens = np.append(lens, values.shape[1:])
    return list(lens)


def _get_place(place):
    place = _get_paddle_place(place)
    if place is None:
        place = _current_expected_place()
    elif not isinstance(
        place, (core.Place, core.CPUPlace, core.CUDAPinnedPlace, core.CUDAPlace)
    ):
        raise ValueError(
            "'place' must be any of paddle.Place, paddle.CPUPlace, paddle.CUDAPinnedPlace, paddle.CUDAPlace"
        )
    return place


def _check_indices_dtype(dtype):
    if dtype not in [paddle.int8, paddle.int16, paddle.int32, paddle.int64]:
        raise TypeError(
            "the dtype of indices must be 'int8' or 'int16' or 'int32' or 'int64'"
        )


def sparse_coo_tensor(
    indices: (
        list[list[int]]
        | tuple[tuple[int, ...], ...]
        | npt.NDArray[np.int_]
        | Tensor
    ),
    values: NumericSequence | npt.NDArray[Any] | Tensor,
    shape: ShapeLike | None = None,
    dtype: DTypeLike | None = None,
    place: CPUPlace | CUDAPinnedPlace | CUDAPlace | str | None = None,
    stop_gradient: bool = True,
) -> Tensor:
    r"""
    Constructs a sparse ``paddle.Tensor`` in coordinate format according to the indices
    and values of the specified non-zero elements.

    Args:
        indices(list|tuple|ndarray|Tensor): the indices of non-zero elements.
            Can be a list, tuple, numpy\.ndarray, paddle\.Tensor. The indices must be 2-D.
        values(list|tuple|ndarray|Tensor): Initial values for the tensor.
            Can be a scalar, list, tuple, numpy\.ndarray, paddle\.Tensor.
        shape(list|tuple|None, optional): The shape of the sparse tensor also represents the shape of
            original dense tensor. If not provided the smallest shape will be inferred to
            hold all elements.
        dtype(str|np.dtype|None, optional): The desired data type of returned tensor. Can be 'bool' , 'float16' ,
            'float32' , 'float64' , 'int8' , 'int16' , 'int32' , 'int64' , 'uint8',
            'complex64' , 'complex128'. Default: None, infers dtype from ``data``
            except for python float number which gets dtype from ``get_default_type`` .
        place(CPUPlace|CUDAPinnedPlace|CUDAPlace|str|None, optional): The place to allocate Tensor. Can be
            CPUPlace, CUDAPinnedPlace, CUDAPlace. Default: None, means global place. If ``place`` is
            string, It can be ``cpu``, ``gpu:x`` and ``gpu_pinned``, where ``x`` is the index of the GPUs.
        stop_gradient(bool, optional): Whether to block the gradient propagation of Autograd. Default: True.

    Returns:
        Tensor: A Tensor constructed from ``indices`` and ``values`` .

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> indices = [[0, 1, 2], [1, 2, 0]]
            >>> values = [1.0, 2.0, 3.0]
            >>> dense_shape = [3, 3]
            >>> coo = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape)
            >>> print(coo)
            Tensor(shape=[3, 3], dtype=paddle.float32, place=Place(cpu), stop_gradient=True,
                   indices=[[0, 1, 2],
                            [1, 2, 0]],
                   values=[1., 2., 3.])
    """

    if in_dynamic_mode():
        place = _get_place(place)
        if not isinstance(indices, core.eager.Tensor):
            indices = to_tensor(
                indices, dtype=None, place=place, stop_gradient=True
            )
        if not isinstance(values, core.eager.Tensor):
            values = to_tensor(values, dtype, place, stop_gradient)
        if len(indices.shape) != 2:
            raise ValueError("'indices' must be 2-D.")

        nnz = indices.shape[1]
        sparse_dim = indices.shape[0]

        _check_indices_dtype(indices.dtype)

        if nnz != values.shape[0]:
            raise ValueError(
                f"the indices and values must have same number of non-zero, but get {nnz} and {values.shape[0]}"
            )

        dense_dim = len(values.shape) - 1

        if not indices.place._equals(place):
            indices = indices._copy_to(place, False)

        if not values.place._equals(place):
            values = values._copy_to(place, False)
        values = _handle_dtype(values, dtype)
        values.stop_gradient = stop_gradient

        min_shape = _infer_dense_shape(indices, values)

        if shape is None:
            shape = min_shape
        else:
            shape = list(shape)
            if shape < min_shape:
                raise ValueError(
                    f"the minimum shape required is {min_shape}, but get {shape}"
                )
            if len(shape) != sparse_dim + dense_dim:
                raise ValueError(
                    f"the number of dimensions(len(shape) must be sparse_dim({sparse_dim}) + dense_dim({dense_dim}), but get {len(shape)}"
                )

        return _C_ops.sparse_sparse_coo_tensor(values, indices, shape)
    elif in_pir_mode():
        return _C_ops.sparse_sparse_coo_tensor(values, indices, shape)
        if shape[0] is None:
            shape[0] = -1
    else:
        op_type = 'sparse_sparse_coo_tensor'
        inputs = {'values': values, 'indices': indices}
        if shape[0] is None:
            shape[0] = -1
        attrs = {'shape': shape}
        helper = LayerHelper(op_type)
        out = helper.create_sparse_variable_for_type_inference(dtype)
        helper.append_op(
            type=op_type, inputs=inputs, outputs={'out': out}, attrs=attrs
        )
        return out


def _infer_dense_csr_shape(crows, cols):
    crows_numpy = crows.numpy()
    cols_numpy = cols.numpy()
    batchs = np.sum(crows_numpy[:-1] > crows_numpy[1:]) + 1
    if (int(len(crows_numpy) / batchs) * batchs) != len(crows_numpy):
        raise ValueError(
            f"The calculated original matrix batch size is {batchs}, but it cannot correctly split the row data. Please carefully check the data or the input shape."
        )
    col = cols_numpy.max() + 1
    row = int(len(crows_numpy) / batchs) - 1
    if batchs == 1:
        return [row, col]
    else:
        return [batchs, row, col]


# TODO: need to support shape is None
@dygraph_only
def sparse_csr_tensor(
    crows: list[int] | tuple[int, ...] | npt.NDArray[np.int_] | Tensor,
    cols: list[int] | tuple[int, ...] | npt.NDArray[np.int_] | Tensor,
    values: NumericSequence | npt.NDArray[Any] | Tensor,
    shape: ShapeLike | None = None,
    dtype: DTypeLike | None = None,
    place: CPUPlace | CUDAPinnedPlace | CUDAPlace | str | None = None,
    stop_gradient: bool = True,
) -> Tensor:
    r"""
    Constructs a sparse ``paddle.Tensor`` in CSR(Compressed Sparse Row) format according to the
    ``crows``, ``cols`` and ``values``.
    Currently, the crows and cols of each batch must be incremented.

    Args:
        crows(list|tuple|ndarray|Tensor): 1-D array, each element in the rows represents the
            starting position of the first non-zero element of each row in values.
            Can be a list, tuple, numpy\.ndarray, paddle\.Tensor.
        cols(list|tuple|ndarray|Tensor): 1-D array, the column of non-zero elements.
            Can be a list, tuple, numpy\.ndarray, paddle\.Tensor.
        values(list|tuple|ndarray|Tensor): 1-D array, the non-zero elements.
            Can be a scalar, list, tuple, numpy\.ndarray, paddle\.Tensor.
        shape(list|tuple, optional): The shape of the sparse tensor also represents the shape of
            original dense tensor.
            hold all elements.
        dtype(str|np.dtype|None, optional): The desired data type of returned tensor. Can be 'bool' , 'float16' ,
            'float32' , 'float64' , 'int8' , 'int16' , 'int32' , 'int64' , 'uint8',
            'complex64' , 'complex128'. Default: None, infers dtype from ``data``
            except for python float number which gets dtype from ``get_default_type`` .
        place(CPUPlace|CUDAPinnedPlace|CUDAPlace|str|None, optional): The place to allocate Tensor. Can be
            CPUPlace, CUDAPinnedPlace, CUDAPlace. Default: None, means global place. If ``place`` is
            string, It can be ``cpu``, ``gpu:x`` and ``gpu_pinned``, where ``x`` is the index of the GPUs.
        stop_gradient(bool, optional): Whether to block the gradient propagation of Autograd. Default: True.

    Returns:
        Tensor: A Tensor constructed from ``crows``, ``cols`` and ``values`` .

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> crows = [0, 2, 3, 5]
            >>> cols = [1, 3, 2, 0, 1]
            >>> values = [1, 2, 3, 4, 5]
            >>> dense_shape = [3, 4]
            >>> csr = paddle.sparse.sparse_csr_tensor(crows, cols, values, dense_shape)
            >>> print(csr)
            Tensor(shape=[3, 4], dtype=paddle.int64, place=Place(cpu), stop_gradient=True,
                   crows=[0, 2, 3, 5],
                   cols=[1, 3, 2, 0, 1],
                   values=[1, 2, 3, 4, 5])
    """

    place = _get_place(place)

    if not isinstance(crows, core.eager.Tensor):
        crows = to_tensor(crows, dtype=None, place=place, stop_gradient=True)
    if not isinstance(cols, core.eager.Tensor):
        cols = to_tensor(cols, dtype=None, place=place, stop_gradient=True)
    if not isinstance(values, core.eager.Tensor):
        values = to_tensor(values, dtype, place, stop_gradient)

    _check_indices_dtype(crows.dtype)
    _check_indices_dtype(cols.dtype)

    if shape is not None:
        if len(shape) != 2 and len(shape) != 3:
            raise ValueError(
                f"SparseCsrTensor only support 2-D or 3-D matrix. but get shape {shape}"
            )
    else:
        shape = _infer_dense_csr_shape(crows, cols)

    rows = shape[len(shape) - 2]

    if not crows.place._equals(place):
        crows = crows._copy_to(place, False)

    if not cols.place._equals(place):
        cols = cols._copy_to(place, False)

    if not values.place._equals(place):
        values = values._copy_to(place, False)
    values = _handle_dtype(values, dtype)
    values.stop_gradient = stop_gradient

    if len(crows.shape) != 1 or len(cols.shape) != 1 or len(values.shape) != 1:
        raise ValueError("The 'crows', 'cols' and 'values' must be 1-D.")

    if len(cols) != len(values):
        raise ValueError("the length of cols must be same as length of values")

    if len(shape) == 2:
        if crows.shape[0] != rows + 1:
            raise ValueError(
                f"The length({crows.shape[0]}) of crows must be equal to the rows({rows})+1 of matrix."
            )
        if crows[0] != 0:
            raise ValueError("the 0th value of crows must be 0")

        if crows[-1] != values.shape[0]:
            raise ValueError(
                "the last value of crows must be equal the number of non-zero"
            )
    else:
        if crows.shape[0] % (rows + 1) != 0:
            raise ValueError(
                f"The length({crows.shape[0]}) of crows must be divisible the rows({rows})+1 of matrix."
            )
    # TODO(zkh2016): check whether the value in crows and cols is legal

    return core.eager.sparse_csr_tensor(
        crows, cols, values, shape, stop_gradient
    )
