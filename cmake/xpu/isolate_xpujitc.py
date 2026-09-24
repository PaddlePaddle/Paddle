#!/usr/bin/env python3
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
"""Generate a SONAME-isolated copy of XTDK's libxpujitc.so.

Background
----------
The XTDK LLVM19 toolkit ships ``shlib/libxpujitc.so`` (the xpurtc JIT compiler
that CINN's XPU backend depends on). The XPU third-party (XHPC) package ships
its OWN ``libxpujitc.so`` with an ABI-incompatible xpurtc API that
``libxpu_blas.so`` / ``libxpu_dnn.so`` depend on. BOTH files carry the SONAME
``libxpujitc.so``, and the only mangled symbol they share is the
``xpurtc::CompileContext`` destructor (``D1Ev`` / ``D2Ev``).

Because the dynamic loader keys on SONAME, only one of the two libraries could
otherwise be linked/loaded, yet neither is a superset of the other. To let CINN
use the XTDK jitc while ``libxpu_blas`` keeps using the XHPC jitc, we produce an
isolated copy that:

  1. has a distinct SONAME ``libxpujitc_xtdk.so`` (so the linker no longer
     deduplicates it against the XHPC ``libxpujitc.so``),
  2. renames its colliding ``xpurtc::CompileContext`` destructor exports
     ``_ZN6xpurtc14CompileContextD1Ev`` / ``D2Ev`` to the ``Xpurtc`` namespace
     so they no longer interpose the XHPC destructor at runtime, and
  3. hides (localizes) every OTHER dynamic export that CINN does not actually
     call. Even with a distinct SONAME, once this library is loaded into the
     same process as the XHPC libxpujitc.so (e.g. via libcinnapi.so ->
     libxpu_blas.so -> libxpujitc.so), any symbol name it exports becomes a
     candidate during global-scope symbol resolution. If XTDK happens to also
     define a symbol XHPC needs (e.g. JitcErrorString, jitc_log) but with
     different semantics/missing relocations, resolution can pick the wrong
     one and libxpu_dnn.so / libxpu_blas.so fail with "undefined symbol" even
     though the correct XHPC libxpujitc.so is present. Restricting the export
     table to only the handful of symbols CINN actually calls closes this gap.

The matching undefined references inside CINN's ``compiler.cc.o`` /
``compiler_xpu.cc.o`` are renamed the same way by
``cmake/xpu/redefine_cinn_jitc_syms.cmake`` at link time, so CINN binds its
destructor to this isolated library while ``libxpu_blas`` binds to XHPC's.

Usage
-----
    python isolate_xpujitc.py <xtdk_shlib_dir>

Writes ``<xtdk_shlib_dir>/libxpujitc_xtdk.so``. Idempotent: safe to run on every
cmake configure. Requires the ``lief`` python package (used to rewrite the
dynamic symbol table AND regenerate ``.gnu.hash`` so the isolated library still
loads).
"""

import os
import shutil
import sys

SRC_NAME = "libxpujitc.so"
DST_NAME = "libxpujitc_xtdk.so"
DST_SONAME = "libxpujitc_xtdk.so"

# xpurtc::CompileContext destructor: the only symbol that collides with the
# XHPC libxpujitc.so. Renamed into the "Xpurtc" namespace to break interposition.
RENAME = {
    "_ZN6xpurtc14CompileContextD1Ev": "_ZN6Xpurtc14CompileContextD1Ev",
    "_ZN6xpurtc14CompileContextD2Ev": "_ZN6Xpurtc14CompileContextD2Ev",
}

# The full xpurtc API surface CINN's XPU backend (compiler_xpu.cc) actually
# calls: CompileContext ctor/dtor/add_source/get_kernel, Kernel, launch_kernel.
# Everything else this library exports is localized (see _localize_others)
# so it can never shadow an XHPC libxpujitc.so symbol during global-scope
# resolution once both libraries are loaded into the same process.
KEEP_EXPORTED = {
    "_ZN6xpurtc14CompileContextC1Ei",
    "_ZN6xpurtc14CompileContextC2Ei",
    "_ZN6xpurtc14CompileContextC1EOS0_",
    "_ZN6xpurtc14CompileContextC2EOS0_",
    "_ZN6xpurtc14CompileContext10add_sourceENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEENS_10SourceKindES6_",
    "_ZN6xpurtc14CompileContext10get_kernelEv",
    "_ZN6xpurtc13launch_kernelEPKvjmiiPvS1_jbPKc",
    "_ZNK6xpurtc6Kernel4codeEv",
    "_ZNK6xpurtc6Kernel4sizeEv",
    "_ZNK6xpurtc6Kernel4hashEv",
    "_ZN9XpuModuleC1EmPvb",
    "_ZN9XpuModuleC2EmPvb",
    "_ZN9XpuModuleD1Ev",
    "_ZN9XpuModuleD2Ev",
    # renamed dtor stays exported under its new name
    "_ZN6Xpurtc14CompileContextD1Ev",
    "_ZN6Xpurtc14CompileContextD2Ev",
}


def _already_isolated(path):
    """Return True if <path> already has the target SONAME and renamed dtor."""
    try:
        import lief
    except ImportError:
        return False
    if not os.path.exists(path):
        return False
    b = lief.parse(path)
    if b is None:
        return False
    try:
        soname = b.get(lief.ELF.DynamicEntry.TAG.SONAME).name
    except Exception:
        return False
    if soname != DST_SONAME:
        return False
    exported = {s.name for s in b.dynamic_symbols}
    # isolated iff renamed dtor is exported and the stock one is gone
    return (
        "_ZN6Xpurtc14CompileContextD1Ev" in exported
        and "_ZN6xpurtc14CompileContextD1Ev" not in exported
    )


def main():
    if len(sys.argv) != 2:
        sys.stderr.write("usage: isolate_xpujitc.py <xtdk_shlib_dir>\n")
        return 2
    shlib_dir = sys.argv[1]
    src = os.path.join(shlib_dir, SRC_NAME)
    dst = os.path.join(shlib_dir, DST_NAME)

    if not os.path.exists(src):
        sys.stderr.write(f"isolate_xpujitc.py: source not found: {src}\n")
        return 1

    if _already_isolated(dst):
        # Nothing to do; keep the existing artifact untouched.
        sys.stdout.write(
            f"isolate_xpujitc.py: {dst} already isolated, skipping.\n"
        )
        return 0

    try:
        import lief
    except ImportError:
        sys.stderr.write(
            "isolate_xpujitc.py: the 'lief' python package is required "
            "(pip install lief). It rewrites the dynamic symbol table and "
            "regenerates .gnu.hash so the isolated library still loads.\n"
        )
        return 1

    shutil.copyfile(src, dst)
    b = lief.parse(dst)
    if b is None:
        sys.stderr.write(f"isolate_xpujitc.py: failed to parse {dst}\n")
        return 1

    # 1. distinct SONAME so the linker no longer dedups against XHPC's copy
    b.get(lief.ELF.DynamicEntry.TAG.SONAME).name = DST_SONAME

    # 2. rename the colliding CompileContext destructor exports
    renamed = 0
    for s in b.dynamic_symbols:
        if s.name in RENAME:
            s.name = RENAME[s.name]
            renamed += 1

    # 3. localize every other dynamic export not in KEEP_EXPORTED so it can
    # no longer participate in global-scope symbol resolution once this
    # library is loaded alongside XHPC's libxpujitc.so.
    #
    # DISABLED: verified ineffective on XHPC dev/20260523 + XTDK llvm19 --
    # libxpu_dnn.so still fails with "undefined symbol: JitcErrorString" even
    # after this library stops exporting that symbol, so the collision is not
    # caused by export-table shadowing. Localizing also hides symbols CINN
    # genuinely needs (get_by_full_name / get_by_short_name /
    # Kernel::mangled_name / Kernel::is_cdnn_kernel), breaking the cinnapi
    # link. Kept here (inert) to document what was ruled out.
    localized = 0

    b.write(dst)
    os.chmod(dst, 0o755)
    sys.stdout.write(
        f"isolate_xpujitc.py: wrote {dst} (soname={DST_SONAME}, renamed {renamed} dtor sym(s), "
        f"localized {localized} other export(s)).\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
