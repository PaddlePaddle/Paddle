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

from .tensor.linalg import cholesky  # noqa: F401
from .tensor.linalg import norm  # noqa: F401
from .tensor.linalg import eig  # noqa: F401
from .tensor.linalg import cov  # noqa: F401
from .tensor.linalg import cond  # noqa: F401
from .tensor.linalg import matrix_power  # noqa: F401
from .tensor.linalg import solve  # noqa: F401
from .tensor.linalg import cholesky_solve  # noqa: F401
from .tensor import inverse as inv  # noqa: F401
from .tensor.linalg import eigvals  # noqa: F401
from .tensor.linalg import multi_dot  # noqa: F401
from .tensor.linalg import matrix_rank  # noqa: F401
from .tensor.linalg import svd  # noqa: F401
from .tensor.linalg import eigvalsh  # noqa: F401
from .tensor.linalg import qr  # noqa: F401
from .tensor.linalg import lu  # noqa: F401
from .tensor.linalg import lu_unpack  # noqa: F401
from .tensor.linalg import eigh  # noqa: F401
from .tensor.linalg import det  # noqa: F401
from .tensor.linalg import slogdet  # noqa: F401
from .tensor.linalg import pinv  # noqa: F401
from .tensor.linalg import triangular_solve  # noqa: F401
from .tensor.linalg import lstsq

__all__ = [
    'cholesky',  #noqa
    'norm',
    'cond',
    'cov',
    'inv',
    'eig',
    'eigvals',
    'multi_dot',
    'matrix_rank',
    'svd',
    'qr',
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
    'lstsq'
]
