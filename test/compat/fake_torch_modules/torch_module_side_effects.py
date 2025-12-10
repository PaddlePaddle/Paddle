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

"""
PyTorch module side effects simulation.

The PyTorch module can only be initialized once. If imported multiple times,
it raises a RuntimeError to simulate side effects during imports.

For example:

    >>> import sys
    >>> import torch  # Works fine the first time
    >>> del sys.modules['torch']
    >>> import torch  # Raises an error due to re-initialization

We simulate this behavior to test how Paddle's torch proxy handles such side effects.
Since the torch module will be cleared when removed from sys.modules, we need to create
a new module to record the initialization state.
"""

initialized = False


def _ensure_init_once():
    global initialized
    if not initialized:
        initialized = True
        return
    raise RuntimeError(
        "torch core module is already initialized, this can happen due to side effects of imports."
    )
