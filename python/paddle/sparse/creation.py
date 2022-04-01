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
from ..tensor import to_tensor
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


#TODO: To support the shape is None
@dygraph_only
def sparse_coo_tensor(indices,
                      values,
                      shape,
                      dtype=None,
                      place=None,
                      stop_gradient=True):
    if not isinstance(indices, core.eager.Tensor):
        indices = to_tensor(
            indices, dtype=None, place=place, stop_gradient=True)
    if not isinstance(values, core.eager.Tensor):
        values = to_tensor(values, dtype, place, stop_gradient)
    if place is not None:
        indices = indices._copy_to(place, False)
        values = values._copy_to(place, False)
    values = _handle_dtype(values, dtype)
    return core.eager.sparse_coo_tensor(indices, values, shape, stop_gradient)


#TODO: To support the shape is None
@dygraph_only
def sparse_csr_tensor(crows,
                      cols,
                      values,
                      shape,
                      dtype=None,
                      place=None,
                      stop_gradient=True):
    if not isinstance(crows, core.eager.Tensor):
        crows = to_tensor(crows, dtype=None, place=place, stop_gradient=True)
    if not isinstance(cols, core.eager.Tensor):
        cols = to_tensor(cols, dtype=None, place=place, stop_gradient=True)
    if not isinstance(values, core.eager.Tensor):
        values = to_tensor(values, dtype, place, stop_gradient)

    if place is not None:
        crows = crows._copy_to(place, False)
        cols = cols._copy_to(place, False)
        values = values._copy_to(place, False)
    values = _handle_dtype(values, dtype)
    return core.eager.sparse_csr_tensor(crows, cols, values, shape,
                                        stop_gradient)
