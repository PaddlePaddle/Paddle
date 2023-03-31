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


import numpy as np

# TODO: define framework api
from paddle.fluid.layer_helper_base import LayerHelperBase

__all__ = []


def set_default_dtype(d):
    """
    Set default dtype. The default dtype is initially float32.

    Args:
        d(string|np.dtype): the dtype to make the default. It only
                            supports float16, bfloat16, float32 and float64.

    Returns:
        None.

    Examples:
        .. code-block:: python

            import paddle
            paddle.set_default_dtype("float32")

    """
    if isinstance(d, type):
        # This branch is for NumPy scalar types
        if d in [np.float16, np.float32, np.float64]:
            d = d.__name__
        else:
            raise TypeError(
                "set_default_dtype only supports [float16, float32, float64] "
                ", but received %s" % d.__name__
            )
    else:
        # This branch is for np.dtype and str
        if d in ['float16', 'float32', 'float64', 'bfloat16']:
            # NOTE(SigureMo): Since the np.dtype object is not an instance of
            # type, so it will not be handled by the previous branch. We need
            # to convert it to str here.
            d = str(d)
        else:
            raise TypeError(
                "set_default_dtype only supports [float16, float32, float64, bfloat16] "
                ", but received %s" % str(d)
            )

    LayerHelperBase.set_default_dtype(d)


def get_default_dtype():
    """
    Get the current default dtype. The default dtype is initially float32.

    Args:
        None.
    Returns:
        String, this global dtype only supports float16, float32, float64.

    Examples:
        .. code-block:: python

            import paddle
            paddle.get_default_dtype()
    """
    return LayerHelperBase.get_default_dtype()
