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

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from packaging.version import InvalidVersion, Version

from paddle.version import full_version as internal_version

__all__ = ["PaddleVersion"]


class PaddleVersion(str):
    """A string with magic powers to compare to both Version and iterables!

    Mirrors ``torch.torch_version.TorchVersion``. ``paddle.__version__`` was
    historically a plain ``str`` and existing user code compares against it
    as a ``str``; ``PaddleVersion`` masquerades as a ``str`` so those keep
    working while also comparing as a ``packaging.version.Version`` against
    another ``Version`` / ``str`` / iterable such as ``(2, 6, 1)``.

    Examples:
        Comparing a PaddleVersion object to a Version object
            PaddleVersion('2.6.0a') > Version('2.6.0a')
        Comparing a PaddleVersion object to a Tuple object
            PaddleVersion('2.6.0a') > (2, 5)    # 2.5
            PaddleVersion('2.6.0a') > (2, 5, 1) # 2.5.1
        Comparing a PaddleVersion object against a string
            PaddleVersion('2.6.0a') > '2.5'
            PaddleVersion('2.6.0a') > '2.5.1'
    """

    __slots__ = ()

    # fully qualified type names here to appease mypy
    def _convert_to_version(self, inp: Any) -> Any:
        if isinstance(inp, Version):
            return inp
        elif isinstance(inp, str):
            return Version(inp)
        elif isinstance(inp, Iterable):
            # Ideally this should work for most cases by attempting to group
            # the version tuple, assuming the tuple looks (MAJOR, MINOR, ?PATCH)
            # Examples:
            #   * (1)         -> Version("1")
            #   * (1, 20)     -> Version("1.20")
            #   * (1, 20, 1)  -> Version("1.20.1")
            return Version(".".join(str(item) for item in inp))
        else:
            raise InvalidVersion(inp)

    def _cmp_wrapper(self, cmp: Any, method: str) -> bool:
        try:
            return getattr(Version(self), method)(self._convert_to_version(cmp))
        except BaseException as e:
            if not isinstance(e, InvalidVersion):
                raise
            # Fall back to regular string comparison if dealing with an invalid
            # version like 'parrot'
            return getattr(super(), method)(cmp)


for cmp_method in ["__gt__", "__lt__", "__eq__", "__ge__", "__le__"]:
    setattr(
        PaddleVersion,
        cmp_method,
        lambda x, y, method=cmp_method: x._cmp_wrapper(y, method),
    )

__version__ = PaddleVersion(internal_version)
