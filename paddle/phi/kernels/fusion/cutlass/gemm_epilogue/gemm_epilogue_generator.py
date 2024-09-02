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

import enum
import os

from gemm_epilogue_common import (
    CommonCutlassGemmEpilogueKernelArguments,
    CommonCutlassGemmEpilogueKernelDeclare,
    CommonCutlassGemmEpilogueKernelExecute,
    CommonGemmEpilogueFunction,
    CommonTail,
    GenerateFunctionForPhi,
)
from util import SubstituteTemplate, TileDesc, parse_args, write_kernel_to_file

build_dir = os.path.abspath(os.path.dirname(__file__)) + '/build/'

mutex_include = '''#include <mutex>'''

fba_header = '''
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/epilogue/thread/linear_combination_leaky_relu.h"
#include "cutlass/epilogue/thread/linear_combination_silu.h"
#include "cutlass/epilogue/thread/linear_combination_bias_relu.h"
#include "cutlass/epilogue/thread/linear_combination_sigmoid.h"
#include "paddle/phi/kernels/fusion/cutlass/gemm_epilogue/fast_gelu.h"
#include "cutlass/util/device_memory.h"
#include "paddle/phi/kernels/fusion/cutlass/gemm_epilogue/gemm_epilogue_util.h"

namespace phi{
namespace fusion{
namespace cutlass_internal{
'''


dict_for_declare_part = {
    "epi_part": "${epi_func}<${element_c}, ${epilogue_vector_length}, ${element_accum}, ${element_epilogue}, ${scale_type}>",
}

fba_kernel = (
    SubstituteTemplate(
        CommonCutlassGemmEpilogueKernelDeclare, dict_for_declare_part
    )
    + CommonCutlassGemmEpilogueKernelArguments
    + CommonCutlassGemmEpilogueKernelExecute
)

fba_kernel_leaky_alpha = fba_kernel.replace(
    "{1.f, 1.f}", "{1.f, 1.f, params.leaky_alpha}"
)


# these three acts are not supported by pass now and are commented out to prevent the lib.so too large.
class FbaAct(enum.Enum):
    Identity = 1
    Relu = 2
    Gelu = 3
    # LeakyRelu = 4
    # Sigmoid = 5
    # Silu = 6


SupportedAct = [
    FbaAct.Identity,
    FbaAct.Relu,
    FbaAct.Gelu,
    # FbaAct.LeakyRelu,
    # FbaAct.Sigmoid,
    # FbaAct.Silu,
]

ActTag = {
    SupportedAct[0]: 'cutlass::epilogue::thread::LinearCombination',
    SupportedAct[1]: 'cutlass::epilogue::thread::LinearCombinationRelu',
    SupportedAct[2]: 'cutlass::epilogue::thread::LinearCombinationFastGELU',
    # SupportedAct[3]: 'cutlass::epilogue::thread::LinearCombinationLeakyRelu',
    # SupportedAct[4]: 'cutlass::epilogue::thread::LinearCombinationSigmoid',
    # SupportedAct[5]: 'cutlass::epilogue::thread::LinearCombinationSilu',
}

UnderScoreName = {
    SupportedAct[0]: "matmul_add",
    SupportedAct[1]: "matmul_add_relu",
    SupportedAct[2]: "matmul_add_gelu",
    # SupportedAct[3]: "matmul_add_leaky_relu",
    # SupportedAct[4]: "matmul_add_sigmoid",
    # SupportedAct[5]: "matmul_add_silu",
}

CamelName = {
    SupportedAct[0]: "MatmulAdd",
    SupportedAct[1]: "MatmulAddRelu",
    SupportedAct[2]: "MatmulAddGelu",
    # SupportedAct[3]: "MatmulAddLeakyRelu",
    # SupportedAct[4]: "MatmulAddSigmoid",
    # SupportedAct[5]: "MatmulAddSilu",
}

# layouts = [
#     (cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor),
#     (cutlass::layout::ColumnMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor),
#     (cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor),
#     (cutlass::layout::RowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor),
# ]
layouts = [
    (
        'cutlass::layout::RowMajor',
        'cutlass::layout::RowMajor',
        'cutlass::layout::RowMajor',
    )
]

swizzling_functors = [
    'GemmIdentityThreadblockSwizzle<1>',
    'GemmIdentityThreadblockSwizzle<2>',
    'GemmIdentityThreadblockSwizzle<4>',
    # 'GemmIdentityThreadblockSwizzle<8>',
]

# (mode == GemmUniversalMode::kGemm) the tile-splitting factor (1 defaults to StreamK, >1 emulates Split-K)
split_k_factors = ["1", "2", "4"]  # ,"8","16"]

alignments = [8]


def generate_sm75_1688():
    kernel_dict = {
        "element_a": "cutlass::half_t",
        "element_b": "cutlass::half_t",
        "element_c": "cutlass::half_t",
        "opcode_class": "cutlass::arch::OpClassTensorOp",
        "arch": "cutlass::arch::Sm75",
        "stages": "2",
        # alpha is always float!
        "element_epilogue": "float",
        "math_operator": "cutlass::arch::OpMultiplyAdd",
        "scale_type": "cutlass::epilogue::thread::ScaleType::NoBetaScaling",
    }
    # The following three parameters need to be the same as alignments
    kernel_dict["epilogue_vector_length"] = "8"
    kernel_dict["align_a"] = "8"
    kernel_dict["align_b"] = "8"
    math_instructions = [
        ("16,8,8", "cutlass::half_t", "cutlass::half_t", "float")
    ]
    kernel_dict["split_k_factor"] = "1"

    sm75_code = ""
    for epi_func in SupportedAct:
        op_dict = {}
        op_dict["func_name"] = UnderScoreName[epi_func].lower() + "_sm75_fp16"
        op_dict["enum_op_name"] = UnderScoreName[epi_func].upper()

        all_kernel_names = ""
        all_kernel_declares = ""
        kernel_dict["epi_func"] = ActTag[epi_func]
        suffix = 0
        for alignment in alignments:
            for swizzling_func in swizzling_functors:
                kernel_dict["swizzling_functor"] = swizzling_func
                if swizzling_func == 'ThreadblockSwizzleStreamK':
                    kernel_dict["split_k_factor"] = "1"
                for layout in layouts:
                    kernel_dict["layout_a"] = layout[0]
                    kernel_dict["layout_b"] = layout[1]
                    kernel_dict["layout_c"] = layout[2]
                    for math_inst in math_instructions:
                        tiles = [
                            TileDesc(
                                "256, 128, 32", 2, "64, 64, 32", math_inst
                            ),
                            TileDesc(
                                "128, 256, 32", 2, "64, 64, 32", math_inst
                            ),
                            TileDesc(
                                "128, 128, 32", 2, "64, 64, 32", math_inst
                            ),
                            TileDesc(
                                " 64, 256, 32", 2, "64, 64, 32", math_inst
                            ),
                            TileDesc(
                                "256,  64, 32", 2, "64, 64, 32", math_inst
                            ),
                            TileDesc(
                                " 64, 128, 32", 2, "32, 64, 32", math_inst
                            ),
                            TileDesc(
                                "128,  64, 32", 2, "64, 32, 32", math_inst
                            ),
                            TileDesc(
                                " 64,  64, 32", 2, "32, 32, 32", math_inst
                            ),
                            TileDesc(
                                " 64, 128, 64", 2, "64, 64, 32", math_inst
                            ),
                        ]
                        for tile in tiles:
                            kernel_dict["Tshape"] = tile.Tshape
                            kernel_dict["Wshape"] = tile.Wshape
                            kernel_dict["Ishape"] = tile.math_inst[0]
                            kernel_dict["element_accum"] = tile.math_inst[3]
                            kernel_dict["kernel_func_name"] = (
                                op_dict["func_name"] + "_" + str(suffix)
                            )
                            suffix += 1

                            fba_kernel_ = fba_kernel
                            # if epi_func in [FbaAct.LeakyRelu]:
                            #     fba_kernel_ = fba_kernel_leaky_alpha
                            kernel_str = (
                                fba_header
                                + SubstituteTemplate(fba_kernel_, kernel_dict)
                                + CommonTail
                            )
                            file_name = (
                                build_dir
                                + "generated_tmp/"
                                + kernel_dict["kernel_func_name"]
                                + ".cu"
                            )
                            write_kernel_to_file(kernel_str, file_name)

                            all_kernel_names += (
                                kernel_dict["kernel_func_name"] + ", \n"
                            )
                            all_kernel_declares += (
                                "cutlass::Status "
                                + kernel_dict["kernel_func_name"]
                                + "(const GemmEpilogueAllParams& params);"
                                + "\n"
                            )

        # Generate op code
        op_dict["kernel_func_declare"] = all_kernel_declares
        op_dict["all_kernel_func_name"] = all_kernel_names
        sm75_code += SubstituteTemplate(CommonGemmEpilogueFunction, op_dict)
    return sm75_code


def sm80_16816_forStreamK(op_dict, kernel_dict, suffix, epi_func):
    all_kernel_names = ""
    all_kernel_declares = ""
    kernel_dict["swizzling_functor"] = 'ThreadblockSwizzleStreamK'
    kernel_dict["split_k_factor"] = "1"
    kernel_dict["kernel_func_name"] = op_dict["func_name"] + "_" + str(suffix)
    suffix += 1

    fba_kernel_ = fba_kernel
    # if epi_func in [FbaAct.LeakyRelu]:
    #     fba_kernel_ = fba_kernel_leaky_alpha
    kernel_str = (
        fba_header + SubstituteTemplate(fba_kernel_, kernel_dict) + CommonTail
    )
    file_name = (
        build_dir + "generated_tmp/" + kernel_dict["kernel_func_name"] + ".cu"
    )
    write_kernel_to_file(kernel_str, file_name)

    all_kernel_names += kernel_dict["kernel_func_name"] + ", \n"
    all_kernel_declares += (
        "cutlass::Status "
        + kernel_dict["kernel_func_name"]
        + "(const GemmEpilogueAllParams& params);"
        + "\n"
    )
    return all_kernel_names, all_kernel_declares, suffix


def sm80_16816_forUniversal(op_dict, kernel_dict, suffix, epi_func):
    all_kernel_names = ""
    all_kernel_declares = ""
    for swizzling_functor in swizzling_functors:
        kernel_dict["swizzling_functor"] = swizzling_functor
        for split_k_fac in split_k_factors:
            kernel_dict["split_k_factor"] = split_k_fac
            kernel_dict["kernel_func_name"] = (
                op_dict["func_name"] + "_" + str(suffix)
            )
            suffix += 1

            fba_kernel_ = fba_kernel
            # if epi_func in [FbaAct.LeakyRelu]:
            #     fba_kernel_ = fba_kernel_leaky_alpha
            kernel_str = (
                fba_header
                + SubstituteTemplate(fba_kernel_, kernel_dict)
                + CommonTail
            )
            file_name = (
                build_dir
                + "generated_tmp/"
                + kernel_dict["kernel_func_name"]
                + ".cu"
            )
            write_kernel_to_file(kernel_str, file_name)

            all_kernel_names += kernel_dict["kernel_func_name"] + ", \n"
            all_kernel_declares += (
                "cutlass::Status "
                + kernel_dict["kernel_func_name"]
                + "(const GemmEpilogueAllParams& params);"
                + "\n"
            )
    return all_kernel_names, all_kernel_declares, suffix


def generate_sm80_16816(cutlass_dtype="cutlass::half_t"):
    kernel_dict = {
        "element_a": cutlass_dtype,
        "element_b": cutlass_dtype,
        "element_c": cutlass_dtype,
        "opcode_class": "cutlass::arch::OpClassTensorOp",
        "arch": "cutlass::arch::Sm80",
        # alpha is always float!
        "element_epilogue": "float",
        "math_operator": "cutlass::arch::OpMultiplyAdd",
        "scale_type": "cutlass::epilogue::thread::ScaleType::NoBetaScaling",
    }
    # The following three parameters need to be the same as alignments
    kernel_dict["epilogue_vector_length"] = "8"
    kernel_dict["align_a"] = "8"
    kernel_dict["align_b"] = "8"

    math_instructions = [("16,8,16", cutlass_dtype, cutlass_dtype, "float")]

    sm80_code = ""
    for epi_func in SupportedAct:
        op_dict = {}
        op_dict["func_name"] = (
            UnderScoreName[epi_func].lower()
            + "_sm80_"
            + ("fp16" if "half" in cutlass_dtype else "bf16")
        )
        op_dict["enum_op_name"] = UnderScoreName[epi_func].upper()

        all_kernel_names = ""
        all_kernel_declares = ""
        kernel_dict["epi_func"] = ActTag[epi_func]
        suffix = 0
        for alignment in alignments:
            for layout in layouts:
                kernel_dict["layout_a"] = layout[0]
                kernel_dict["layout_b"] = layout[1]
                kernel_dict["layout_c"] = layout[2]
                for math_inst in math_instructions:
                    tiles = [
                        TileDesc("256, 128, 32", 3, "64, 64, 32", math_inst),
                        TileDesc("128, 256, 32", 3, "64, 64, 32", math_inst),
                        TileDesc("256, 64, 32", 3, "64, 64, 32", math_inst),
                        TileDesc("256, 64, 32", 4, "64, 64, 32", math_inst),
                        TileDesc("64, 256, 32", 4, "64, 64, 32", math_inst),
                        TileDesc("128, 128, 32", 3, "64, 64, 32", math_inst),
                        TileDesc("128, 128, 32", 4, "64, 64, 32", math_inst),
                        TileDesc("128, 128, 32", 5, "64, 64, 32", math_inst),
                        TileDesc("128, 64, 32", 6, "64, 32, 32", math_inst),
                        TileDesc("64, 128, 32", 6, "32, 64, 32", math_inst),
                        TileDesc("64, 64, 32", 10, "32, 32, 32", math_inst),
                        TileDesc("256, 128, 64", 3, "64, 64, 64", math_inst),
                        TileDesc("128, 256, 64", 3, "64, 64, 64", math_inst),
                        TileDesc("256, 64, 64", 4, "64, 64, 64", math_inst),
                        TileDesc("64, 256, 64", 4, "64, 64, 64", math_inst),
                        TileDesc("128, 128, 64", 4, "64, 64, 64", math_inst),
                        TileDesc("256, 64, 64", 3, "64, 64, 64", math_inst),
                        TileDesc("64, 256, 64", 3, "64, 64, 64", math_inst),
                        TileDesc("128, 128, 64", 3, "64, 64, 64", math_inst),
                        TileDesc("128, 64, 64", 3, "64, 32, 64", math_inst),
                        TileDesc("64, 128, 64", 3, "32, 64, 64", math_inst),
                        TileDesc("64, 64, 64", 5, "32, 32, 64", math_inst),
                        #
                        TileDesc("32, 64, 64", 5, "16, 32, 64", math_inst),
                        TileDesc("16, 64, 64", 5, "16, 32, 64", math_inst),
                    ]
                    for tile in tiles:
                        kernel_dict["Tshape"] = tile.Tshape
                        kernel_dict["Wshape"] = tile.Wshape
                        kernel_dict["Ishape"] = tile.math_inst[0]
                        kernel_dict["stages"] = str(tile.stages)
                        kernel_dict["element_accum"] = tile.math_inst[3]
                        (
                            all_kernel_names_universal,
                            all_kernel_declares_universal,
                            suffix,
                        ) = sm80_16816_forUniversal(
                            op_dict, kernel_dict, suffix, epi_func
                        )
                        # all_kernel_names_streamk, all_kernel_declares_streamk, suffix = sm80_16816_forStreamK(op_dict, kernel_dict, suffix, epi_func)
                        all_kernel_declares += all_kernel_declares_universal
                        # all_kernel_declares += all_kernel_declares_streamk
                        all_kernel_names += all_kernel_names_universal
                        # all_kernel_names += all_kernel_names_streamk
        # Generate op code
        op_dict["kernel_func_declare"] = all_kernel_declares
        op_dict["all_kernel_func_name"] = all_kernel_names
        sm80_code += SubstituteTemplate(CommonGemmEpilogueFunction, op_dict)
    return sm80_code


if __name__ == "__main__":
    sm_versions_and_types = []
    args = parse_args()
    all_code = mutex_include + fba_header
    # fp32 is not considered for the time being
    if args.cuda_arch == "75":
        sm_versions_and_types.append(["75", "fp16"])
        all_code += generate_sm75_1688()
    if args.cuda_arch in ["80", "86", "89"]:
        sm_versions_and_types.append(["80", "fp16"])
        sm_versions_and_types.append(["80", "bf16"])
        all_code += generate_sm80_16816()
        all_code += generate_sm80_16816(cutlass_dtype="cutlass::bfloat16_t")

    all_code += GenerateFunctionForPhi(
        sm_versions_and_types, SupportedAct, UnderScoreName, CamelName
    )
    all_code += CommonTail
    with open(build_dir + "generated_tmp/matmul_add_act.cu", "w") as f:
        f.write(all_code)
