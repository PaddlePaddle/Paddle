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

from .tensor import inverse as inv  # noqa: F401
from .tensor.linalg import (
    cholesky,  # noqa: F401
    cholesky_solve,  # noqa: F401
    cond,  # noqa: F401
    corrcoef,  # noqa: F401
    cov,  # noqa: F401
    det,  # noqa: F401
    eig,  # noqa: F401
    eigh,  # noqa: F401
    eigvals,  # noqa: F401
    eigvalsh,  # noqa: F401
    householder_product,  # noqa: F401
    lstsq,
    lu,  # noqa: F401
    lu_unpack,  # noqa: F401
    matrix_power,  # noqa: F401
    matrix_rank,  # noqa: F401
    multi_dot,  # noqa: F401
    norm,  # noqa: F401
    pca_lowrank,  # noqa: F401
    pinv,  # noqa: F401
    qr,  # noqa: F401
    slogdet,  # noqa: F401
    solve,  # noqa: F401
    svd,  # noqa: F401
    triangular_solve,  # noqa: F401
)

__all__ = [
    'cholesky',
    'norm',
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
    'lu',
    'lu_unpack',
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
]
