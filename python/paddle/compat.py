# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved
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
#
# This file implements most of the public API compatible with PyTorch.
# Note that this file does not depend on PyTorch in any way.
# This is a standalone implementation.

import sys
import warnings
from contextlib import contextmanager

from .tensor.compat import (
    Unfold,
    max,
    median,
    min,
    nanmedian,
    sort,
    split,
)
from .tensor.compat_softmax import softmax


class TorchMetaFinder:
    """
    PyTorch compatibility layer for PaddlePaddle.

    This class provides a way to `import torch` but actually loads PaddlePaddle.

    Inspired by the setuptools _distutils_hack.
    """

    def find_spec(self, fullname, path, target=None):
        if fullname != "torch" and not fullname.startswith("torch."):
            return None

        import importlib
        import importlib.abc
        import importlib.util

        module_name = fullname.replace("torch", "paddle", 1)
        module = importlib.import_module(module_name)

        class TorchLoader(importlib.abc.Loader):
            def create_module(self, spec):
                module.__name__ = 'torch'
                return module

            def exec_module(self, module):
                pass

        return importlib.util.spec_from_loader(
            'torch', TorchLoader(), origin=module.__file__
        )


TORCH_FINDER = TorchMetaFinder()


def install_torch_alias():
    sys.meta_path.insert(0, TORCH_FINDER)


def uninstall_torch_alias():
    if TORCH_FINDER in sys.meta_path:
        sys.meta_path.remove(TORCH_FINDER)
        if 'torch' in sys.modules:
            del sys.modules['torch']
        return
    warnings.warn("torch alias is not installed.")


@contextmanager
def enable_torch_alias_guard():
    install_torch_alias()
    try:
        yield
    finally:
        uninstall_torch_alias()


__all__ = [
    'softmax',
    'split',
    'sort',
    'Unfold',
    'min',
    'max',
    'median',
    'nanmedian',
]
