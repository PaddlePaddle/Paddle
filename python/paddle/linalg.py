# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved
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

from .tensor import inverse as inv
from .tensor.linalg import (
    cholesky,
    cholesky_solve,
    cond,
    corrcoef,
    cov,
    det,
    eig,
    eigh,
    eigvals,
    eigvalsh,
    householder_product,
    lstsq,
    lu,
    lu_unpack,
    matrix_exp,
    matrix_norm,
    matrix_power,
    matrix_rank,
    multi_dot,
    norm,
    ormqr,
    pca_lowrank,
    pinv,
    qr,
    slogdet,
    solve,
    svd,
    svd_lowrank,
    triangular_solve,
    vector_norm,
)

__all__ = [
    'cholesky',
    'norm',
    'matrix_norm',
    'vector_norm',
    'cond',
    'cov',
    'corrcoef',
    'inv',
    'eig',
    'eigvals',
    'multi_dot',
    'matrix_rank',
    'svd',
    'qr',
    'householder_product',
    'pca_lowrank',
    'svd_lowrank',
    'lu',
    'lu_unpack',
    'matrix_exp',
    'matrix_power',
    'det',
    'slogdet',
    'eigh',
    'eigvalsh',
    'pinv',
    'solve',
    'cholesky_solve',
    'triangular_solve',
    'lstsq',
    'ormqr',
]
