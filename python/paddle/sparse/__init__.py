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
from ..framework import core
from ..fluid.framework import _in_legacy_dygraph, in_dygraph_mode, _in_eager_without_dygraph_check
from ..tensor import to_tensor

__all__ = [
    'sparse_coo_tensor',
    'sparse_csr_tensor',
]


#TODO: need to support the shape is None
@dygraph_only
def sparse_coo_tensor(indices,
                      values,
                      shape,
                      dtype=None,
                      place=None,
                      stop_gradient=True):
    #Sparse API can only work in eager mode
    if not isinstance(data, core.eager.Tensor):
        indices = to_tensor(
            indices, dtype=None, place=place, stop_gradient=True)
        values = to_tensor(values, dtype, place, stop_gradient)
    if _in_eager_without_dygraph_check():
        return core.eager.sparse_coo_tensor(indices, values, shape,
                                            stop_gradient)


#TODO: need to support the shape is None
@dygraph_only
def sparse_csr_tensor(crows,
                      cols,
                      values,
                      shape,
                      dtype=None,
                      place=None,
                      stop_gradient=True):
    #Sparse API can only work in eager mode
    crows = to_tensor(crows, dtype=None, place=place, stop_gradient=True)
    cols = to_tensor(cols, dtype=None, place=place, stop_gradient=True)
    values = to_tensor(values, dtype, place, stop_gradient)
    if _in_eager_without_dygraph_check():
        return core.eager.sparse_csr_tensor(indices, values, shape,
                                            stop_gradient)
