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

import ap


class CompileCommandGenerator:
    def __init__(self):
        self.file_ext = "cu"
        self.op_type2generate_func = ap.OrderedDict(
            [
                ['matmul', self.generate_matmul_compile_command],
            ]
        )

    def __call__(self, op_type, tpl_dirname, library_name):
        return self.op_type2generate_func[op_type](tpl_dirname, library_name)

    def generate_matmul_compile_command(self, tpl_dirname, library_name):
        matmul_source_dir = f"{tpl_dirname}/matmul"

        compile_cmd = "nvcc -std=c++20 -O3 -Xcompiler=-fPIC -arch=sm_80 --expt-relaxed-constexpr"
        compile_cmd = compile_cmd + " -I ${AP_CUTLASS_DIR}/include"
        compile_cmd = compile_cmd + " -I ${AP_CUTLASS_DIR}/tools/util/include"
        compile_cmd = compile_cmd + " -I " + matmul_source_dir
        compile_cmd = (
            compile_cmd
            + " -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0"
        )
        compile_cmd = (
            compile_cmd + " -DAP_ENABLE_AUTOTUNE=1 -DAP_ENABLE_DEBUG=0"
        )
        compile_cmd = (
            compile_cmd
            + f" --shared {library_name}.{self.file_ext} -o lib{library_name}.so"
        )
        return compile_cmd
