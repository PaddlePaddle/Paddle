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

"""
Check module boundary rules to prevent dependency regression.

Usage:
    python ci/check_module_boundary.py [--verbose]

This script verifies that lower-layer modules do not include headers from
upper-layer modules, enforcing the architectural layering defined in:
    review/modules/design.md

Exit code 0 means no violations; non-zero means violations found.
"""

import argparse
import glob
import os
import re
import sys

# Each rule: (source_glob, forbidden_include_regex, reason)
RULES = [
    (
        "paddle/phi/core/**/*.[ch]*",
        r'#include\s+"paddle/fluid/',
        "phi/core must not depend on fluid",
    ),
    (
        "paddle/phi/kernels/**/*.[ch]*",
        r'#include\s+"paddle/fluid/',
        "phi/kernels must not depend on fluid",
    ),
    (
        "paddle/pir/**/*.[ch]*",
        r'#include\s+"paddle/fluid/',
        "pir must not depend on fluid",
    ),
    (
        "paddle/cinn/**/*.[ch]*",
        r'#include\s+"paddle/fluid/framework/',
        "cinn must not depend on fluid/framework",
    ),
    (
        "paddle/fluid/framework/**/*.[ch]*",
        r'#include\s+"paddle/fluid/operators/',
        "framework must not depend on operators",
    ),
    (
        "paddle/common/**/*.[ch]*",
        r'#include\s+"paddle/phi/',
        "common must not depend on phi",
    ),
    (
        "paddle/common/**/*.[ch]*",
        r'#include\s+"paddle/fluid/',
        "common must not depend on fluid",
    ),
]

# Known exceptions that are allowed temporarily (will be fixed later).
# Format: (relative_file_path, line_content_substring)
EXCEPTIONS = [
    # Phase 2.4: inference engine ops need interface injection (future work)
    ("paddle/fluid/framework/naive_executor.cc", "tensorrt_engine_op.h"),
    ("paddle/fluid/framework/naive_executor.cc", "openvino_engine_op.h"),
    # control_flow_op_helper.h depends on assign_op.h (complex, deferred)
    (
        "paddle/fluid/framework/new_executor/interpreter/static_build.cc",
        "control_flow_op_helper.h",
    ),
    # ops_signature is tightly coupled with operator.cc (needs separate fix)
    ("paddle/fluid/framework/operator.cc", "ops_signature/signatures.h"),
    # nccl_gpu_common.h used for var_type_traits (needs separate fix)
    ("paddle/fluid/framework/var_type_traits.cc", "nccl_gpu_common.h"),
    # Phase 3: pir -> fluid (to be fixed when PIR is promoted to top-level)
    ("paddle/pir/include/pass/pass_registry.h", "paddle/fluid/pir/drr/"),
    # Phase 3: cinn -> fluid/framework (to be fixed in Phase 3)
    (
        "paddle/cinn/hlir/dialect/operator/transforms/"
        "pir_to_py_code_converter.cc",
        "paddle/fluid/framework/",
    ),
    (
        "paddle/cinn/ir/group_schedule/search/measurer.h",
        "paddle/fluid/framework/",
    ),
]

# Files to skip (comments, forwarding headers, etc.)
SKIP_PATTERNS = [
    r"\.pyc$",
    r"\.pyo$",
]


def _is_exception(rel_path, line):
    """Check if a violation is in the known exceptions list."""
    for exc_path, exc_substr in EXCEPTIONS:
        if rel_path == exc_path and exc_substr in line:
            return True
    return False


def find_violations(root, rules, verbose=False):
    """Scan source files and report rule violations."""
    violations = []

    for source_glob, forbidden_re, reason in rules:
        pattern = os.path.join(root, source_glob)
        files = glob.glob(pattern, recursive=True)

        for filepath in files:
            if any(re.search(sp, filepath) for sp in SKIP_PATTERNS):
                continue
            if not os.path.isfile(filepath):
                continue

            try:
                with open(
                    filepath, "r", encoding="utf-8", errors="ignore"
                ) as f:
                    for lineno, line in enumerate(f, 1):
                        # Skip comments (simple heuristic: lines starting with
                        # // or within /* */ blocks that mention paths)
                        stripped = line.strip()
                        if stripped.startswith("//"):
                            continue
                        if stripped.startswith("*"):
                            continue
                        if re.search(forbidden_re, line):
                            rel_path = os.path.relpath(filepath, root)
                            if _is_exception(rel_path, line):
                                continue
                            violations.append(
                                (rel_path, lineno, line.strip(), reason)
                            )
                            if verbose:
                                print(
                                    f"  VIOLATION: {rel_path}:{lineno}: "
                                    f"{line.strip()}"
                                )
                                print(f"    Reason: {reason}")
            except OSError:
                continue

    return violations


def main():
    parser = argparse.ArgumentParser(description="Check module boundary rules")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print each violation"
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Repository root (default: auto-detect from script location)",
    )
    args = parser.parse_args()

    if args.root:
        root = args.root
    else:
        # Script is at ci/check_module_boundary.py, root is parent
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if not os.path.isdir(os.path.join(root, "paddle")):
        print(f"ERROR: Cannot find 'paddle/' directory under root: {root}")
        sys.exit(2)

    print(f"Checking module boundaries in: {root}")
    print(f"Rules: {len(RULES)}")
    print()

    violations = find_violations(root, RULES, verbose=args.verbose)

    if violations:
        print(f"\nFOUND {len(violations)} VIOLATION(S):")
        if not args.verbose:
            for rel_path, lineno, line, reason in violations:
                print(f"  {rel_path}:{lineno}: {line}")
                print(f"    -> {reason}")
        sys.exit(1)
    else:
        print("OK: No module boundary violations found.")
        sys.exit(0)


if __name__ == "__main__":
    main()
