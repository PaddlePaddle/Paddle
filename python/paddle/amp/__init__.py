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

from .auto_cast import (  # noqa: F401
    auto_cast,
    decorate,
    amp_guard,
    amp_decorate,
)
from .amp_lists import (  # noqa: F401
    white_list,
    black_list,
)

from . import (  # noqa: F401
    debugging,
    grad_scaler,
    accuracy_compare,
)

from .grad_scaler import (  # noqa: F401
    GradScaler,
    AmpScaler,
    OptimizerState,
)

from paddle.base import core
from paddle.base.framework import (
    _current_expected_place,
    _get_paddle_place,
)

__all__ = [
    'auto_cast',
    'GradScaler',
    'decorate',
    'is_float16_supported',
    'is_bfloat16_supported',
]


def is_float16_supported(device=None):
    """
    Determine whether the place supports float16 in the auto-mixed-precision training.

    Args:
        device (str|None, optional): Specify the running device.
            It can be ``cpu``, ``gpu``, ``xpu``, ``gpu:x`` and ``xpu:x``,
            where ``x`` is the index of the GPUs or XPUs. if device is None, the device is the current device. Default: None.

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> paddle.amp.is_float16_supported() # True or False
            False
    """

    device = (
        _current_expected_place()
        if device is None
        else _get_paddle_place(device)
    )

    return core.is_float16_supported(device)


def is_bfloat16_supported(device=None):
    """
    Determine whether the place supports bfloat16 in the auto-mixed-precision training.

    Args:
        device (str|None, optional): Specify the running device.
            It can be ``cpu``, ``gpu``, ``xpu``, ``gpu:x`` and ``xpu:x``,
            where ``x`` is the index of the GPUs or XPUs. if device is None, the device is the current device. Default: None.

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> paddle.amp.is_bfloat16_supported() # True or False
            True
    """

    device = (
        _current_expected_place()
        if device is None
        else _get_paddle_place(device)
    )

    return core.is_bfloat16_supported(device)
