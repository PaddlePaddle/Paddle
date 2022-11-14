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

# TODO: define framework api
from paddle.fluid.layer_helper_base import LayerHelperBase
from paddle.fluid.framework import _dygraph_tracer
import numpy as np
from contextlib import contextmanager

__all__ = []


def set_default_dtype(d):
    """
    Set default dtype. The default dtype is initially float32.

    Args:
        d(string|np.dtype): the dtype to make the default. It only
                            supports float16, float32 and float64.

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
        if d in ['float16', 'float32', 'float64']:
            # NOTE(SigureMo): Since the np.dtype object is not an instance of
            # type, so it will not be handled by the previous branch. We need
            # to convert it to str here.
            d = str(d)
        else:
            raise TypeError(
                "set_default_dtype only supports [float16, float32, float64] "
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


@contextmanager
def set_grad_enabled(mode):
    """
    Create a context which enables or disables dygraph gradient calculation.

    Args:
        mode(bool): whether to enable (`True`), or disable (`False`) grad.

    Examples:
        .. code-block:: python

            import paddle
            x = paddle.ones([3, 2])
            x.stop_gradient = False
            with paddle.set_grad_enabled(False):
                y = x * 2
                with paddle.set_grad_enabled(True):
                    z = x * 2
            print(y.stop_gradient)   # True
            print(z.stop_gradient)   # False
    """

    tracer = _dygraph_tracer()
    if tracer:
        prev_mode = tracer._has_grad
        tracer._has_grad = mode
        try:
            yield
        finally:
            tracer._has_grad = prev_mode
    else:
        yield


def is_grad_enabled():
    """
    Returns whether current dygraph gradient calculation mode is enabled.

    Returns:
        bool: True if current dygraph gradient calculation mode is enabled, otherwise false.

    Examples:
        .. code-block:: python

            import paddle

            # Dygraph gradient calculation mode is enabled by default.
            paddle.is_grad_enabled() # True

            with paddle.set_grad_enabled(False):
                paddle.is_grad_enabled() # False

            paddle.enable_static()
            paddle.is_grad_enabled() # False
    """
    tracer = _dygraph_tracer()
    return tracer._has_grad if tracer else False
