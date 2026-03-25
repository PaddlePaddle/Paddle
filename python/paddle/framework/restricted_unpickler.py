# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
Restricted Unpickler for secure deserialization of model files.

This module provides a RestrictedUnpickler that only allows a whitelist
of safe classes to be deserialized, preventing arbitrary code execution
via malicious pickle payloads (CWE-502).
"""

from __future__ import annotations

import pickle

# Whitelist of allowed modules and their allowed classes.
# Only these classes can be instantiated during deserialization.
_ALLOWED_CLASSES: dict[str, set[str]] = {
    # NumPy types (required for model parameters)
    'numpy': {
        'ndarray',
        'dtype',
        'float32',
        'float64',
        'float16',
        'int32',
        'int64',
        'int16',
        'int8',
        'uint8',
        'bool_',
        'complex64',
        'complex128',
        'bfloat16',
    },
    'numpy.core.multiarray': {
        '_reconstruct',
        'scalar',
    },
    'numpy.core.numeric': {
        '*',
    },
    'numpy._core.multiarray': {
        '_reconstruct',
        'scalar',
    },
    'numpy._core.numeric': {
        '*',
    },
    # Collections (required for state_dict structures)
    'collections': {
        'OrderedDict',
        'defaultdict',
    },
    # Python builtins (required for basic data types in state dicts)
    'builtins': {
        'dict',
        'list',
        'tuple',
        'set',
        'frozenset',
        'bytes',
        'bytearray',
        'str',
        'int',
        'float',
        'bool',
        'complex',
        'slice',
        'range',
        'type',
    },
    # copyreg (used by pickle protocol for reconstructing objects)
    'copyreg': {
        '_reconstructor',
    },
    # _codecs (used for encoding in pickle)
    '_codecs': {
        'encode',
    },
    # Paddle internal: safe DenseTensor reconstruction function
    'paddle.framework.io_utils': {
        '_reconstruct_dense_tensor_data',
    },
    # Paddle internal: generator state for RNG serialization
    'paddle.base.libpaddle': {
        'GeneratorState',
    },
    # Paddle internal: distributed flex checkpoint metadata classes
    # These dataclasses are serialized via paddle.save() during checkpoint
    # operations and must be allowed for paddle.load() to restore them.
    'paddle.distributed.flex_checkpoint.dcp.metadata': {
        'Metadata',
        'LocalTensorMetadata',
        'LocalTensorIndex',
    },
}


class RestrictedUnpickler(pickle.Unpickler):
    """A restricted unpickler that only allows whitelisted classes.

    This prevents arbitrary code execution during deserialization by
    blocking dangerous modules such as os, subprocess, builtins.eval,
    builtins.exec, etc.

    Usage:
        with open('model.pdparams', 'rb') as f:
            data = RestrictedUnpickler(f).load()
    """

    def find_class(self, module: str, name: str) -> type:
        """Override find_class to restrict which classes can be loaded.

        Args:
            module: The module name containing the class.
            name: The class name to load.

        Returns:
            The class object if it is in the whitelist.

        Raises:
            pickle.UnpicklingError: If the class is not in the whitelist.
        """
        allowed_names = _ALLOWED_CLASSES.get(module)
        if allowed_names is not None:
            if '*' in allowed_names or name in allowed_names:
                return super().find_class(module, name)

        raise pickle.UnpicklingError(
            f"Forbidden class: {module}.{name}. "
            f"For security, only whitelisted classes are allowed during "
            f"deserialization of model files. If you believe this class "
            f"should be allowed, please report an issue at "
            f"https://github.com/PaddlePaddle/Paddle/issues"
        )


def safe_load_pickle(f, encoding='latin1'):
    """Safely load a pickle file using RestrictedUnpickler.

    Args:
        f: A file-like object (opened in binary mode) to read from.
        encoding: The encoding to use for unpickling (default: 'latin1').

    Returns:
        The deserialized Python object.

    Raises:
        pickle.UnpicklingError: If the pickle data contains forbidden classes.
    """
    return RestrictedUnpickler(f, encoding=encoding).load()


def safe_loads_pickle(data, encoding='latin1'):
    """Safely load pickle data from bytes using RestrictedUnpickler.

    Args:
        data: Bytes or bytearray containing pickled data.
        encoding: The encoding to use for unpickling (default: 'latin1').

    Returns:
        The deserialized Python object.

    Raises:
        pickle.UnpicklingError: If the pickle data contains forbidden classes.
    """
    import io

    return RestrictedUnpickler(io.BytesIO(data), encoding=encoding).load()
