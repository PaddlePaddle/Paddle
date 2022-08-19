# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from typing import Union

import numpy as np

__all__ = ["pcm16to32", "depth_convert"]


def pcm16to32(audio: np.ndarray) -> np.ndarray:
    """pcm int16 to float32

    Args:
        audio (np.ndarray): Waveform with dtype of int16.

    Returns:
        np.ndarray: Waveform with dtype of float32.
    """
    if audio.dtype == np.int16:
        audio = audio.astype("float32")
        bits = np.iinfo(np.int16).bits
        audio = audio / (2**(bits - 1))
    return audio


def _safe_cast(y: np.ndarray, dtype: Union[type, str]) -> np.ndarray:
    """Data type casting in a safe way, i.e., prevent overflow or underflow.

    Args:
        y (np.ndarray): Input waveform array in 1D or 2D.
        dtype (Union[type, str]): Data type of waveform.

    Returns:
        np.ndarray: `y` after safe casting.
    """
    if 'float' in str(y.dtype):
        return np.clip(y, np.finfo(dtype).min,
                       np.finfo(dtype).max).astype(dtype)
    else:
        return np.clip(y, np.iinfo(dtype).min,
                       np.iinfo(dtype).max).astype(dtype)


def depth_convert(y: np.ndarray, dtype: Union[type, str]) -> np.ndarray:
    """Convert audio array to target dtype safely. 
    This function convert audio waveform to a target dtype, with addition steps of
    preventing overflow/underflow and preserving audio range.

    Args:
        y (np.ndarray): Input waveform array in 1D or 2D.
        dtype (Union[type, str]): Data type of waveform.

    Returns:
        np.ndarray: `y` after safe casting.
    """

    SUPPORT_DTYPE = ['int16', 'int8', 'float32', 'float64']
    if y.dtype not in SUPPORT_DTYPE:
        raise ParameterError(
            'Unsupported audio dtype, '
            f'y.dtype is {y.dtype}, supported dtypes are {SUPPORT_DTYPE}')

    if dtype not in SUPPORT_DTYPE:
        raise ParameterError(
            'Unsupported audio dtype, '
            f'target dtype  is {dtype}, supported dtypes are {SUPPORT_DTYPE}')

    if dtype == y.dtype:
        return y

    if dtype == 'float64' and y.dtype == 'float32':
        return _safe_cast(y, dtype)
    if dtype == 'float32' and y.dtype == 'float64':
        return _safe_cast(y, dtype)

    if dtype == 'int16' or dtype == 'int8':
        if y.dtype in ['float64', 'float32']:
            factor = np.iinfo(dtype).max
            y = np.clip(y * factor, np.iinfo(dtype).min,
                        np.iinfo(dtype).max).astype(dtype)
            y = y.astype(dtype)
        else:
            if dtype == 'int16' and y.dtype == 'int8':
                factor = np.iinfo('int16').max / np.iinfo('int8').max - EPS
                y = y.astype('float32') * factor
                y = y.astype('int16')

            else:  # dtype == 'int8' and y.dtype=='int16':
                y = y.astype('int32') * np.iinfo('int8').max / \
                    np.iinfo('int16').max
                y = y.astype('int8')

    if dtype in ['float32', 'float64']:
        org_dtype = y.dtype
        y = y.astype(dtype) / np.iinfo(org_dtype).max
    return y
