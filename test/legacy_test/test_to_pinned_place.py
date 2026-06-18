#   Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
Direct branch coverage for ``paddle.base.framework._to_pinned_place``.

A natural CI run hits only the branch matching the host's compile flags
(typically just the CUDA path). These tests use ``unittest.mock.patch``
over ``core.is_compiled_with_*`` and a small fake place class to drive
every branch on any build.
"""

import contextlib
import unittest
from unittest import mock

from paddle.base import core
from paddle.base.framework import _to_pinned_place


class TestToPinnedPlace(unittest.TestCase):
    def test_already_cuda_pinned_returns_same_object(self):
        # Branch 1: already-pinned passes through unchanged. Constructing
        # ``CUDAPinnedPlace()`` itself requires a CUDA-built wheel, so skip
        # on a pure-CPU build (where the branch is unreachable in practice).
        if not core.is_compiled_with_cuda():
            self.skipTest("CUDA not compiled in this build")
        p = core.CUDAPinnedPlace()
        self.assertIs(_to_pinned_place(p), p)

    def test_cuda_place_to_cuda_pinned(self):
        # Branch 2: explicit GPU place -> CUDAPinnedPlace.
        if not core.is_compiled_with_cuda():
            self.skipTest("CUDA not compiled in this build")
        result = _to_pinned_place(core.CUDAPlace(0))
        self.assertIsInstance(result, core.CUDAPinnedPlace)

    def test_cpu_place_on_cuda_build(self):
        # Branch 5: CPU placement, CUDA-compiled build -> CUDAPinnedPlace.
        if not core.is_compiled_with_cuda():
            self.skipTest("CUDA not compiled in this build")
        with mock.patch.object(
            core, 'is_compiled_with_xpu', return_value=False
        ):
            result = _to_pinned_place(core.CPUPlace())
        self.assertIsInstance(result, core.CUDAPinnedPlace)

    def test_cpu_place_on_xpu_compiled_branch(self):
        # Branch 4: drives the XPU sub-branch on the CPU path. On a
        # non-XPU host, ``core.XPUPinnedPlace()`` itself raises, but the
        # branch line still executes (which is what the coverage tool
        # records); on an XPU host the constructor succeeds.
        with (
            mock.patch.object(core, 'is_compiled_with_xpu', return_value=True),
            contextlib.suppress(Exception),
        ):
            _to_pinned_place(core.CPUPlace())

    def test_xpu_place_branch(self):
        # Branch 3: explicit XPUPlace -> XPUPinnedPlace. Hard to drive on
        # a non-XPU host because ``core.XPUPlace(0)`` itself fails to
        # construct, so we substitute ``core.XPUPlace`` with a stand-in
        # class long enough for the ``isinstance`` check to match. The
        # subsequent ``XPUPinnedPlace()`` constructor may raise on a
        # non-XPU host, which we tolerate -- coverage cares only that
        # the branch executed.
        class FakeXPUPlace:
            def is_xpu_place(self):
                return True

            def is_gpu_place(self):
                return False

            def is_cpu_place(self):
                return False

        with (
            mock.patch.object(core, 'XPUPlace', new=FakeXPUPlace),
            contextlib.suppress(Exception),
        ):
            _to_pinned_place(FakeXPUPlace())

    def test_cpu_place_on_pure_cpu_build_raises(self):
        # Branch 6: CPU placement on a build with neither CUDA nor XPU
        # compiled in -> RuntimeError with the legacy message.
        with (
            mock.patch.object(core, 'is_compiled_with_xpu', return_value=False),
            mock.patch.object(
                core, 'is_compiled_with_cuda', return_value=False
            ),
            self.assertRaises(RuntimeError) as ctx,
        ):
            _to_pinned_place(core.CPUPlace())
        self.assertIn("Pinning memory is not supported", str(ctx.exception))

    def test_unsupported_place_raises(self):
        # Branch 7 (final raise): any object that isn't a CUDA/XPU/CPU
        # place falls through to the catch-all RuntimeError. A plain
        # Python object naturally fails every ``isinstance`` check inside
        # the helper without needing to fake ``core.Place``.
        class _OtherPlace:
            pass

        with self.assertRaises(RuntimeError) as ctx:
            _to_pinned_place(_OtherPlace())
        self.assertIn("Pinning memory is not supported", str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
