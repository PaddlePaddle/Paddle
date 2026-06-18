# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import annotations

import re

import numpy as np

from paddle.base import core


def _print_tensor_in_gpu(tensor):
    """
    Print a GPU tensor's dtype, shape, and all data values directly from
    the device using a single-thread CUDA kernel (device-side printf).

    This function is **CUDA Graph safe**: no host/device memory transfer
    is performed (shape is passed via kernel-argument registers), so it
    can be called inside a CUDA Graph capture region.

    Args:
        tensor (paddle.Tensor): A GPU DenseTensor to print. Must already
            reside on a CUDA device (call ``tensor.cuda()`` first if needed).

    Raises:
        ValueError: If PaddlePaddle is not compiled with CUDA support.
        InvalidArgument: If the tensor is not a DenseTensor or not on GPU.

    Examples:
        .. code-block:: pycon

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.device.set_device('gpu')
            >>> x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
            >>> paddle.utils.gpu_utils._print_tensor_in_gpu(x)

    """
    if not core.is_compiled_with_cuda():
        raise ValueError(
            "paddle.utils._print_tensor_in_gpu is not supported in "
            "CPU-only PaddlePaddle. Please reinstall PaddlePaddle with GPU "
            "support to call this API."
        )
    core.eager._print_tensor_in_gpu(tensor)


# Mapping from the dtype name printed by DebugPrintGPUTensor to a numpy dtype.
_DTYPE_STR_TO_NUMPY = {
    'FLOAT32': np.float32,
    'FLOAT64': np.float64,
    'FLOAT16': np.float16,
    'BFLOAT16': np.float32,  # NumPy has no bfloat16; promote to float32
    'INT32': np.int32,
    'INT64': np.int64,
    'INT16': np.int16,
    'INT8': np.int8,
    'UINT8': np.uint8,
    'BOOL': np.bool_,
}

# Mapping from the dtype name to the paddle dtype string.
_DTYPE_STR_TO_PADDLE = {
    'FLOAT32': 'float32',
    'FLOAT64': 'float64',
    'FLOAT16': 'float16',
    'BFLOAT16': 'bfloat16',
    'INT32': 'int32',
    'INT64': 'int64',
    'INT16': 'int16',
    'INT8': 'int8',
    'UINT8': 'uint8',
    'BOOL': 'bool',
}


def _parse_tensor_from_gpu_print(text):
    """
    Reconstruct a ``paddle.Tensor`` from the text output produced by
    :func:`paddle.utils.gpu_utils._print_tensor_in_gpu`.

    The expected input format is the output written to stdout by the
    ``DebugPrintGPUTensor`` CUDA kernel::

        [TensorDebug] dtype : FLOAT32
        [TensorDebug] shape : [2, 3]
        [TensorDebug] numel : 6
        [TensorDebug] data  :
        [[1, 2, 3],
         [4, 5, 6]]

    Special cases handled:

    * **Scalar (0-D tensor)** – the data line is ``[TensorDebug] data  : <value>``
      (no newline before the value).
    * **Empty tensor (numel == 0)** – the data line ends with ``[]``.

    Args:
        text (str): The captured stdout string from a call to
            ``paddle.utils.gpu_utils._print_tensor_in_gpu``.

    Returns:
        paddle.Tensor: A GPU tensor with the dtype and values recovered from
        *text*.  Call ``.cpu()`` on the result if you need a CPU tensor.

    Raises:
        ValueError: If *text* cannot be parsed (missing header lines, unknown
            dtype, mismatched numel, …).

    Examples:
        .. code-block:: pycon

            >>> import paddle
            >>> import os, sys, tempfile
            >>> # Capture the C-level stdout written by the CUDA printf kernel.
            >>> def capture(tensor):
            ...     with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as f:
            ...         path = f.name
            ...     sys.stdout.flush()
            ...     old = os.dup(1)
            ...     fd = os.open(path, os.O_WRONLY | os.O_TRUNC)
            ...     os.dup2(fd, 1)
            ...     os.close(fd)
            ...     paddle.utils.gpu_utils._print_tensor_in_gpu(tensor)
            ...     paddle.device.synchronize()
            ...     sys.stdout.flush()
            ...     os.dup2(old, 1)
            ...     os.close(old)
            ...     text = open(path).read()
            ...     os.remove(path)
            ...     return text
            >>> x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
            >>> text = capture(x)
            >>> y = paddle.utils.gpu_utils._parse_tensor_from_gpu_print(text)
            >>> print(y)
    """
    import paddle  # local import to avoid circular imports at module load time

    # ------------------------------------------------------------------ #
    # 1. Extract header fields                                             #
    # ------------------------------------------------------------------ #
    dtype_m = re.search(r'\[TensorDebug\] dtype\s*:\s*(\w+)', text)
    shape_m = re.search(r'\[TensorDebug\] shape\s*:\s*\[([^\]]*)\]', text)
    numel_m = re.search(r'\[TensorDebug\] numel\s*:\s*(\d+)', text)
    data_m = re.search(r'\[TensorDebug\] data\s*:(.*)', text, re.DOTALL)

    if not dtype_m:
        raise ValueError(
            "_parse_tensor_from_gpu_print: could not find "
            "'[TensorDebug] dtype' line in the provided text."
        )
    if not shape_m:
        raise ValueError(
            "_parse_tensor_from_gpu_print: could not find "
            "'[TensorDebug] shape' line in the provided text."
        )
    if not numel_m:
        raise ValueError(
            "_parse_tensor_from_gpu_print: could not find "
            "'[TensorDebug] numel' line in the provided text."
        )
    if not data_m:
        raise ValueError(
            "_parse_tensor_from_gpu_print: could not find "
            "'[TensorDebug] data' line in the provided text."
        )

    dtype_str = dtype_m.group(1).strip().upper()
    shape_raw = shape_m.group(1).strip()
    numel = int(numel_m.group(1).strip())
    data_raw = data_m.group(1).strip()

    if dtype_str not in _DTYPE_STR_TO_NUMPY:
        raise ValueError(
            f"_parse_tensor_from_gpu_print: unknown dtype '{dtype_str}'. "
            f"Supported: {list(_DTYPE_STR_TO_NUMPY.keys())}"
        )

    np_dtype = _DTYPE_STR_TO_NUMPY[dtype_str]
    paddle_dtype = _DTYPE_STR_TO_PADDLE[dtype_str]

    # Parse shape: "" means scalar (0-D), otherwise comma-separated ints.
    if shape_raw == '':
        shape = []
    else:
        shape = [int(s.strip()) for s in shape_raw.split(',') if s.strip()]

    # ------------------------------------------------------------------ #
    # 2. Parse data section                                                #
    # ------------------------------------------------------------------ #
    gpu_place = paddle.CUDAPlace(0)

    if numel == 0:
        # Empty tensor – no data values to parse.
        arr = np.empty(shape, dtype=np_dtype)
        t = paddle.to_tensor(arr, dtype=paddle_dtype, place=gpu_place)
        return t

    if len(shape) == 0:
        # Scalar (0-D tensor): data_raw is the single value directly.
        value_str = data_raw
        arr = np.array(_parse_value(value_str, dtype_str), dtype=np_dtype)
        t = paddle.to_tensor(arr, dtype=paddle_dtype, place=gpu_place)
        return t

    # General N-D case: strip all brackets, whitespace, and split by commas.
    # The nested bracket notation produced by PrintTensorKernel uses standard
    # Python list syntax, so ast.literal_eval can reconstruct the nested list
    # directly – but only for numeric dtypes.  For 'BOOL', PrintValue emits
    # "True"/"False" which are valid Python literals too.
    flat_values = _extract_flat_values(data_raw, dtype_str, numel)

    arr = np.array(flat_values, dtype=np_dtype).reshape(shape)
    t = paddle.to_tensor(arr, dtype=paddle_dtype, place=gpu_place)
    return t


def _parse_value(value_str, dtype_str):
    """Parse a single scalar value string from the debug output."""
    value_str = value_str.strip()
    if dtype_str == 'BOOL':
        if value_str == 'True':
            return True
        elif value_str == 'False':
            return False
        else:
            raise ValueError(
                f"_parse_tensor_from_gpu_print: cannot parse bool value "
                f"'{value_str}'."
            )
    return float(value_str)


def _extract_flat_values(data_raw, dtype_str, numel):
    """
    Extract a flat list of Python scalars from the nested-bracket string
    produced by PrintTensorKernel.

    Strategy: remove all bracket characters and split the remaining string
    on commas / whitespace to get individual token strings, then convert
    each token to the appropriate scalar type.
    """
    # Remove all '[' and ']' characters.
    flat_str = data_raw.replace('[', '').replace(']', '')

    # Split on commas (values are separated by ", " or ",\n indent").
    tokens = re.split(r'[,\s]+', flat_str)
    tokens = [t.strip() for t in tokens if t.strip()]

    if len(tokens) != numel:
        raise ValueError(
            f"_parse_tensor_from_gpu_print: expected {numel} values but "
            f"found {len(tokens)} tokens in the data section. "
            f"Parsed tokens: {tokens[:20]}{'...' if len(tokens) > 20 else ''}"
        )

    if dtype_str == 'BOOL':
        result = []
        for tok in tokens:
            if tok == 'True':
                result.append(True)
            elif tok == 'False':
                result.append(False)
            else:
                raise ValueError(
                    f"_parse_tensor_from_gpu_print: cannot parse bool token "
                    f"'{tok}'."
                )
        return result

    # Integer dtypes must preserve exact values.
    # GPU printf may emit scientific notation (e.g. "1e+06"), so use
    # Decimal for lossless string-to-integer conversion.
    from decimal import Decimal

    _INT_DTYPES = {'INT8', 'INT16', 'INT32', 'INT64', 'UINT8'}
    if dtype_str in _INT_DTYPES:
        return [int(Decimal(t)) for t in tokens]

    return [float(t) for t in tokens]
