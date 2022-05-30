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

from paddle.fluid import core
from .fluid import framework
__all__ = []

_already_switch_to_eager_ = False

if not framework._in_eager_mode_:
    for name in dir(core.ops):
        globals()[name] = getattr(core.ops, name)
        __all__.append(name)
    _already_switch_to_eager_ = False
else:
    for name in dir(core.eager.ops):
        globals()[name] = getattr(core.eager.ops, name)
        __all__.append(name)
    _already_switch_to_eager_ = True


def switch_to_core_ops():
    global _already_switch_to_eager_
    if _already_switch_to_eager_:
        for name in dir(core.eager.ops):
            del globals()[name]
            __all__.remove(name)
        for name in dir(core.ops):
            globals()[name] = getattr(core.ops, name)
            __all__.append(name)
        _already_switch_to_eager_ = False


def switch_to_eager_ops():
    global _already_switch_to_eager_
    if not _already_switch_to_eager_:
        for name in dir(core.ops):
            del globals()[name]
            __all__.remove(name)
        for name in dir(core.eager.ops):
            globals()[name] = getattr(core.eager.ops, name)
            __all__.append(name)
        _already_switch_to_eager_ = True
