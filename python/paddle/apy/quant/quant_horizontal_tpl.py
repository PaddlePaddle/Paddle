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

import ap


class QuantHorizontalTemplate:
    def compile(
        self,
        input0_karg,
        input1_karg,
        output0_karg,
        output1_karg,
        quant_scale0_karg,
        quant_scale1_karg,
    ):
        project_module = self.make_project(
            input0_karg,
            input1_karg,
            output0_karg,
            output1_karg,
            quant_scale0_karg,
            quant_scale1_karg,
        )
        return CodeGenResult(  # noqa: F821
            module=project_module,
            kernel_dispatch_func=KernelDispatch,
            kernel_dispatch_const_data=ap.SerializableAttrMap(
                kernel_args_getters=[
                    input0_karg.runtime_getter,
                    input1_karg.runtime_getter,
                    output0_karg.runtime_getter,
                    output1_karg.runtime_getter,
                    quant_scale0_karg.runtime_getter,
                    quant_scale1_karg.runtime_getter,
                ]
            ),
        )

    def make_project(
        self,
        input0_karg,
        input1_karg,
        output0_karg,
        output1_karg,
        quant_scale0_karg,
        quant_scale1_karg,
    ):
        code = """
extern "C" {
void DualQuantKernel(void* stream_ptr, const float* input0, const float* input1, float* output0, float*scale0, float*outpu1, float*scale1, float*output2) {
}
}
"""
        compile_cmd = "nvcc --compiler-options '-fPIC' --shared dual_quant_kernel.cu -o libdual_quant_kernel.so"

        return CodeModule(  # noqa: F821
            FuncDeclare(  # noqa: F821
                ap.DataType.void,
                "DualQuantKernel",
                [
                    ap.PointerType.void_ptr,
                    ap.PointerType.const_float_ptr,
                    ap.PointerType.const_float_ptr,
                    ap.PointerType.float_ptr,
                    ap.PointerType.float_ptr,
                    ap.PointerType.float_ptr,
                    ap.PointerType.float_ptr,
                ],
            ),
            Project(  # noqa: F821
                nested_files=Project.Directory(  # noqa: F821
                    [
                        "dual_quant_kernel.cu",
                        Project.FileContent(code),  # noqa: F821
                    ],
                    ["make.sh", Project.FileContent(compile_cmd)],  # noqa: F821
                ),
                compile_cmd="sh make.sh",
                so_relative_path="libdual_quant_kernel.so",
            ),
        )


def KernelDispatch(ctx):
    import ap

    so_func = ctx.get_so_function("DualQuantKernel")
    stream_ptr = ctx.device_ctx.get_stream_addr_as_void_ptr()
    getters = ctx.kernel_dispatch_const_data.kernel_args_getters
    args = [stream_ptr, *ap.map(lambda getter: getter(ctx), getters)]
    ap.apply(so_func, args)
