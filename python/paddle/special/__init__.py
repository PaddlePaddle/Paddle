# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

"""
paddle.special module provides special mathematical functions for PyTorch compatibility.
"""


def round(input=None, decimals=0, name=None, *, out=None, x=None):
    """
    Alias for paddle.round for PyTorch compatibility.

    Args:
        input (Tensor, optional): Input tensor. Alias: x.
        decimals (int, optional): Number of decimal places. Default: 0.
        name (str, optional): Name for the operation.
        out (Tensor, optional): Output tensor.

    Returns:
        Tensor: Rounded tensor.
    """
    import paddle

    # Handle parameter aliases
    if input is not None:
        x = input
    if x is None:
        raise ValueError("round() requires 'input' or 'x' argument")
    return paddle.round(x, decimals=decimals, name=name, out=out)


__all__ = ['round']
