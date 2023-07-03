#!/usr/bin/env python3

# Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

import subprocess
import sys


def main():
    path = sys.argv[1]
    out_path = sys.argv[2]
    llvm_config = sys.argv[3]

    srcs = []
    srcs.append('#include <absl/strings/string_view.h>')
    # srcs.append('#include "paddle/cinn/backends/llvm/cinn_runtime_llvm_ir.h"\n')
    srcs.append('namespace cinn::backends {')
    srcs.append("static const absl::string_view kRuntimeLlvmIr(")
    srcs.append('R"ROC(')
    with open(path, 'r') as fr:
        srcs.append(fr.read())

    srcs.append(')ROC"')
    srcs.append(');\n')

    cmd = f"{llvm_config} --version"
    version = (
        subprocess.check_output(cmd, shell=True)
        .decode('utf-8')
        .strip()
        .split('.')
    )
    srcs.append("struct llvm_version {")
    for v, n in zip(["major", "minor", "micro"], version):
        srcs.append(
            "  static constexpr int k{} = {};".format(
                v.title(), ''.join(filter(str.isdigit, n))
            )
        )
    srcs.append("};")

    srcs.append('}  // namespace cinn::backends')
    with open(out_path, 'w') as fw:
        fw.write("\n".join(srcs))


def get_clang_version():
    pass


if __name__ == "__main__":
    main()
