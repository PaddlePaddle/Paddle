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

from paddle import _C_ops
from ..framework import core, dygraph_only
from ..framework import _current_expected_place, _get_paddle_place
from ..tensor import to_tensor
from ..tensor import max
from ..fluid.data_feeder import check_variable_and_dtype, check_type, check_dtype, convert_dtype

__all__ = [
    'sparse_coo_tensor',
    'sparse_csr_tensor',
]


def _handle_dtype(data, dtype):
    if dtype:
        if convert_dtype(dtype) != convert_dtype(data.dtype):
            return data.astype(convert_dtype(dtype))
    return data


def _infer_dense_shape(indices):
    assert len(indices.shape) == 2
    lens = max(indices, axis=1)
    lens = lens + 1
    return list(lens.numpy())


def _get_place(place):
    place = _get_paddle_place(place)
    if place is None:
        place = _current_expected_place()
    elif not isinstance(place, (core.Place, core.CPUPlace, core.CUDAPinnedPlace,
                                core.CUDAPlace)):
        raise ValueError(
            "'place' must be any of paddle.Place, paddle.CPUPlace, paddle.CUDAPinnedPlace, paddle.CUDAPlace"
        )
    return place


@dygraph_only
def sparse_coo_tensor(indices,
                      values,
                      shape=None,
                      dtype=None,
                      place=None,
                      stop_gradient=True):
    r"""
    Constructs a sparse ``paddle.Tensor`` in coordinate format according to the indices 
    and values of the specified non-zero elements.

    Args:
        indices(list|tuple|ndarray|Tensor): the indices of non-zero elements.
            Can be a list, tuple, numpy\.ndarray, paddle\.Tensor. The indices must be 2-D.
        values(list|tuple|ndarray|Tensor): Initial values for the tensor.
            Can be a scalar, list, tuple, numpy\.ndarray, paddle\.Tensor.
        shape(list|tuple, optional): The shape of the sparse tensor also represents the shape of
            original dense tensor. If not provided the smallest shape will be inferred to 
            hold all elements.
        dtype(str|np.dtype, optional): The desired data type of returned tensor. Can be 'bool' , 'float16' , 
            'float32' , 'float64' , 'int8' , 'int16' , 'int32' , 'int64' , 'uint8',
            'complex64' , 'complex128'. Default: None, infers dtype from ``data`` 
            except for python float number which gets dtype from ``get_default_type`` .
        place(CPUPlace|CUDAPinnedPlace|CUDAPlace|str, optional): The place to allocate Tensor. Can be  
            CPUPlace, CUDAPinnedPlace, CUDAPlace. Default: None, means global place. If ``place`` is 
            string, It can be ``cpu``, ``gpu:x`` and ``gpu_pinned``, where ``x`` is the index of the GPUs. 
        stop_gradient(bool, optional): Whether to block the gradient propagation of Autograd. Default: True.

    Returns:
        Tensor: A Tensor constructed from ``indices`` and ``values`` .

    Raises:
        TypeError: If the data type of ``values`` is not list, tuple, numpy.ndarray, paddle.Tensor
        ValueError: If ``values`` is tuple|list, it can't contain nested tuple|list with different lengths , such as: [[1, 2], [3, 4, 5]]. If the ``indices`` is not a 2-D. 
        TypeError: If ``dtype`` is not bool, float16, float32, float64, int8, int16, int32, int64, uint8, complex64, complex128
        ValueError: If ``place`` is not paddle.CPUPlace, paddle.CUDAPinnedPlace, paddle.CUDAPlace or specified pattern string. 

    Examples:

    .. code-block:: python

        import paddle
        from paddle.fluid.framework import _test_eager_guard

        with _test_eager_guard():
            indices = [[0, 1, 2], [1, 2, 0]]
            values = [1.0, 2.0, 3.0]
            dense_shape = [2, 3]
            coo = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape)
            # print(coo)
            # Tensor(shape=[2, 3], dtype=paddle.float32, place=Place(gpu:0), stop_gradient=True,
            #       indices=[[0, 1, 2],
            #                [1, 2, 0]],
            #       values=[1., 2., 3.])
    """

    place = _get_place(place)

    if not isinstance(indices, core.eager.Tensor):
        indices = to_tensor(
            indices, dtype=None, place=place, stop_gradient=True)
    if not isinstance(values, core.eager.Tensor):
        values = to_tensor(values, dtype, place, stop_gradient)
    if len(indices.shape) != 2:
        raise ValueError("'indices' must be 2-D.")

    if not indices.place._equals(place):
        indices = indices._copy_to(place, False)

    if not values.place._equals(place):
        values = values._copy_to(place, False)
    values = _handle_dtype(values, dtype)
    values.stop_gradient = stop_gradient

    if shape is None:
        shape = _infer_dense_shape(indices)

    return _C_ops.final_state_sparse_create_sparse_coo_tensor(values, indices,
                                                              shape)


#TODO: need to support shape is None
@dygraph_only
def sparse_csr_tensor(crows,
                      cols,
                      values,
                      shape,
                      dtype=None,
                      place=None,
                      stop_gradient=True):
    r"""
    Constructs a sparse ``paddle.Tensor`` in CSR(Compressed Sparse Row) format according to the 
    ``crows``, ``cols`` and ``values``.

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
        dtype(str|np.dtype, optional): The desired data type of returned tensor. Can be 'bool' , 'float16' , 
            'float32' , 'float64' , 'int8' , 'int16' , 'int32' , 'int64' , 'uint8',
            'complex64' , 'complex128'. Default: None, infers dtype from ``data`` 
            except for python float number which gets dtype from ``get_default_type`` .
        place(CPUPlace|CUDAPinnedPlace|CUDAPlace|str, optional): The place to allocate Tensor. Can be  
            CPUPlace, CUDAPinnedPlace, CUDAPlace. Default: None, means global place. If ``place`` is 
            string, It can be ``cpu``, ``gpu:x`` and ``gpu_pinned``, where ``x`` is the index of the GPUs. 
        stop_gradient(bool, optional): Whether to block the gradient propagation of Autograd. Default: True.

    Returns:
        Tensor: A Tensor constructed from ``crows``, ``cols`` and ``values`` .

    Raises:
        TypeError: If the data type of ``values`` is not list, tuple, numpy.ndarray, paddle.Tensor
        ValueError: If ``values`` is tuple|list, it can't contain nested tuple|list with different lengths , such as: [[1, 2], [3, 4, 5]]. If the ``crow``, ``cols`` and ``values`` is not a 2-D. 
        TypeError: If ``dtype`` is not bool, float16, float32, float64, int8, int16, int32, int64, uint8, complex64, complex128
        ValueError: If ``place`` is not paddle.CPUPlace, paddle.CUDAPinnedPlace, paddle.CUDAPlace or specified pattern string. 

    Examples:

    .. code-block:: python

        import paddle
        from paddle.fluid.framework import _test_eager_guard

        with _test_eager_guard():
            crows = [0, 2, 3, 5]
            cols = [1, 3, 2, 0, 1]
            values = [1, 2, 3, 4, 5]
            dense_shape = [3, 4]
            csr = paddle.sparse.sparse_csr_tensor(crows, cols, values, dense_shape)
            # print(csr)
            # Tensor(shape=[3, 4], dtype=paddle.int64, place=Place(gpu:0), stop_gradient=True,
            #       crows=[0, 2, 3, 5],
            #       cols=[1, 3, 2, 0, 1],
            #       values=[1, 2, 3, 4, 5])
    """

    place = _get_place(place)

    if not isinstance(crows, core.eager.Tensor):
        crows = to_tensor(crows, dtype=None, place=place, stop_gradient=True)
    if not isinstance(cols, core.eager.Tensor):
        cols = to_tensor(cols, dtype=None, place=place, stop_gradient=True)
    if not isinstance(values, core.eager.Tensor):
        values = to_tensor(values, dtype, place, stop_gradient)
    if len(crows.shape) != 1 or len(cols.shape) != 1 or len(values.shape) != 1:
        raise ValueError(
            "SparseCsrTensor only support 2-D or 3-D matrix. The 'crows', 'cols' and 'values' must be 1-D."
        )

    if not crows.place._equals(place):
        crows = crows._copy_to(place, False)

    if not cols.place._equals(place):
        cols = cols._copy_to(place, False)

    if not values.place._equals(place):
        values = values._copy_to(place, False)
    values = _handle_dtype(values, dtype)
    values.stop_gradient = stop_gradient
    return core.eager.sparse_csr_tensor(crows, cols, values, shape,
                                        stop_gradient)
