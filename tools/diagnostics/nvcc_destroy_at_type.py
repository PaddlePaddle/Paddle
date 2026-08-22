#!/usr/bin/env python3

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

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


def find_stl_construct() -> Path:
    result = subprocess.run(
        ["c++", "-std=c++20", "-E", "-H", "-x", "c++", "-"],
        input="#include <bits/stl_construct.h>\n",
        text=True,
        capture_output=True,
        check=True,
    )
    for line in result.stderr.splitlines():
        candidate = Path(line.lstrip(". "))
        if candidate.as_posix().endswith("/bits/stl_construct.h"):
            return candidate
    raise RuntimeError("c++ did not report the path to bits/stl_construct.h")


def create_overlay(build_dir: Path) -> Path:
    source = find_stl_construct()
    contents = source.read_text()

    namespace_marker = "_GLIBCXX_BEGIN_NAMESPACE_VERSION"
    destructor = "__location->~_Tp();"
    if contents.count(namespace_marker) != 1 or contents.count(destructor) != 1:
        raise RuntimeError(f"unexpected libstdc++ header layout: {source}")

    contents = contents.replace(
        namespace_marker,
        namespace_marker
        + "\n\n  template <typename>\n"
        + "    struct __paddle_nvcc_destroy_at_type;",
        1,
    )
    contents = contents.replace(
        destructor,
        "(void)sizeof(__paddle_nvcc_destroy_at_type<_Tp>);\n\t" + destructor,
        1,
    )

    overlay = build_dir / "nvcc-type-dump"
    output = overlay / "bits/stl_construct.h"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(contents)
    return overlay


def command_args(entry: dict) -> list[str]:
    if "arguments" in entry:
        return entry["arguments"]
    return shlex.split(entry["command"])


def main() -> int:
    if len(sys.argv) < 3:
        print(
            f"usage: {sys.argv[0]} COMPILE_COMMANDS SOURCE [SOURCE ...]",
            file=sys.stderr,
        )
        return 2

    database_path = Path(sys.argv[1]).resolve()
    database = json.loads(database_path.read_text())
    overlay = create_overlay(database_path.parent)
    keep_dir = database_path.parent / "nvcc-keep"
    keep_dir.mkdir(exist_ok=True)

    env = os.environ.copy()
    env["CCACHE_DISABLE"] = "1"
    diagnostic_flags = (
        f"-I{overlay} --keep --keep-dir={keep_dir} "
        "--ftemplate-backtrace-limit=0 --display-error-number"
    )
    env["NVCC_PREPEND_FLAGS"] = " ".join(
        part
        for part in (diagnostic_flags, env.get("NVCC_PREPEND_FLAGS", ""))
        if part
    )

    for source in sys.argv[2:]:
        matches = [
            entry
            for entry in database
            if Path(entry["file"]).as_posix().endswith(source)
        ]
        if len(matches) != 1:
            print(
                f"expected one compile command for {source}, found {len(matches)}",
                file=sys.stderr,
            )
            return 2

        entry = matches[0]
        args = command_args(entry)
        print(f"\n=== NVCC destroy_at diagnostic: {source} ===", flush=True)
        print(shlex.join(args), flush=True)
        result = subprocess.run(args, cwd=entry["directory"], env=env)
        print(f"diagnostic compiler exit code: {result.returncode}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
