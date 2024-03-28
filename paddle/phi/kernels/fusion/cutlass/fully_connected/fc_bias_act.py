import enum

from fc_common import(
    CommonFcFunction,
    CommonCutlassFcKernelDeclare,
    CommonCutlassFcKernelArguments,
    CommonCutlassFcKernelExecute,
    CommonTail,
    GenerateFunctionForPhi
)
from util import SubstituteTemplate, TileDesc, parse_args, write_kernel_to_file

import os
build_dir = os.path.abspath(os.path.dirname(__file__)) + '/build/'

mutex_include = '''#include <mutex>'''

fba_header = '''
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/epilogue/thread/linear_combination_leaky_relu.h"
#include "cutlass/epilogue/thread/linear_combination_silu.h"
#include "cutlass/epilogue/thread/linear_combination_bias_relu.h"
#include "cutlass/epilogue/thread/linear_combination_sigmoid.h"
#include "cutlass/util/device_memory.h"
#include "paddle/phi/kernels/fusion/cutlass/fully_connected/fc_util.h"

namespace phi{
namespace fusion{
namespace cutlass_internal{
'''


# cutlass::epilogue::thread::LinearCombinationRelu<cutlass::half_t, 8, float, float>
dict_for_declare_part = {
    "epi_part": "${epi_func}<${element_c}, ${epilogue_vector_length}, ${element_accum}, ${element_epilogue}, ${scale_type}>",
}

fba_kernel = (
    SubstituteTemplate(CommonCutlassFcKernelDeclare, dict_for_declare_part) 
    + CommonCutlassFcKernelArguments
    + CommonCutlassFcKernelExecute
)

fba_kernel_leaky_alpha = fba_kernel.replace(
    "{1.f, 1.f}", "{1.f, 1.f, params.leaky_alpha}"
)

class FbaAct(enum.Enum):
    Identity = 1
    Relu = 2
    Silu = 3
    LeakyRelu = 4
    Sigmoid = 5
    Gelu = 6

SupportedAct = [
    FbaAct.Identity,
    FbaAct.Relu,
    FbaAct.Silu,
    FbaAct.LeakyRelu,
    FbaAct.Sigmoid,
    FbaAct.Gelu,
]

ActTag = {
    SupportedAct[0]: 'cutlass::epilogue::thread::LinearCombination',
    SupportedAct[1]: 'cutlass::epilogue::thread::LinearCombinationRelu',
    SupportedAct[2]: 'cutlass::epilogue::thread::LinearCombinationSilu',
    SupportedAct[3]: 'cutlass::epilogue::thread::LinearCombinationLeakyRelu',
    SupportedAct[4]: 'cutlass::epilogue::thread::LinearCombinationSigmoid',
    SupportedAct[5]: 'cutlass::epilogue::thread::LinearCombinationGELU',
}

UnderScoreName = {
    SupportedAct[0]: "fc_bias",
    SupportedAct[1]: "fc_bias_relu",
    SupportedAct[2]: "fc_bias_silu",
    SupportedAct[3]: "fc_bias_leaky_relu",
    SupportedAct[4]: "fc_bias_sigmoid",
    SupportedAct[5]: "fc_bias_gelu",
}

CamelName = {
    SupportedAct[0]: "FcBias",
    SupportedAct[1]: "FcBiasRelu",
    SupportedAct[2]: "FcBiasSilu",
    SupportedAct[3]: "FcBiasLeakyRelu",
    SupportedAct[4]: "FcBiasSigmoid",
    SupportedAct[5]: "FcBiasGelu",
}

# layouts = [
#     (cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor),
#     (cutlass::layout::ColumnMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor),
#     (cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor),
#     (cutlass::layout::RowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor),
# ]
layouts = [
    ('cutlass::layout::RowMajor', 'cutlass::layout::RowMajor', 'cutlass::layout::RowMajor')
]

# cutlass 写死是1的 我理解就等同于没有swizzle 或许是它通用的原因？
swizzling_functors = [
    'GemmIdentityThreadblockSwizzle<1>',
    'ThreadblockSwizzleStreamK'
]

# (mode == GemmUniversalMode::kGemm) the tile-splitting factor (1 defaults to StreamK, >1 emulates Split-K)
split_k_factors = ["1","2","4","8","16"]

# 暂时没有考虑fp32 所以只用128bit对齐 128/sizeof(half)
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
    # 以下三个和alignments需要对齐一下
    kernel_dict["epilogue_vector_length"] = "8"
    kernel_dict["align_a"] = "8"
    kernel_dict["align_b"] = "8"
    math_instructions = [
        ("16,8,8", "cutlass::half_t", "cutlass::half_t", "float")
    ]
    # (mode == GemmUniversalMode::kGemm) the tile-splitting factor (1 defaults to StreamK, >1 emulates Split-K)
    kernel_dict["split_k_factor"] = "1"

    sm75_code = ""
    for epi_func in SupportedAct:
        op_dict = {}
        op_dict["func_name"] = (
            UnderScoreName[epi_func].lower() + "_sm75_fp16"
        )
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
                            TileDesc("256, 128, 32", 2, "64, 64, 32", math_inst),
                            TileDesc("128, 256, 32", 2, "64, 64, 32", math_inst),
                            TileDesc("128, 128, 32", 2, "64, 64, 32", math_inst),
                            TileDesc(" 64, 256, 32", 2, "64, 64, 32", math_inst),
                            TileDesc("256,  64, 32", 2, "64, 64, 32", math_inst),
                            TileDesc(" 64, 128, 32", 2, "32, 64, 32", math_inst),
                            TileDesc("128,  64, 32", 2, "64, 32, 32", math_inst),
                            TileDesc(" 64,  64, 32", 2, "32, 32, 32", math_inst),
                            TileDesc(" 64, 128, 64", 2, "64, 64, 32", math_inst),
                        ]
                        for tile in tiles:
                            kernel_dict["Tshape"] = tile.Tshape
                            kernel_dict["Wshape"] = tile.Wshape
                            kernel_dict["Ishape"] = tile.math_inst[0]
                            kernel_dict["element_accum"] = tile.math_inst[3]
                            kernel_dict["kernel_func_name"] = op_dict["func_name"] + "_" + str(suffix)
                            suffix += 1
                            
                            fba_kernel_ = fba_kernel
                            if epi_func in [FbaAct.LeakyRelu]:
                                fba_kernel_ = fba_kernel_leaky_alpha
                            kernel_str = (
                                fba_header + SubstituteTemplate(fba_kernel_, kernel_dict) + CommonTail
                            )
                            file_name = (
                                build_dir + "generated_tmp/" + kernel_dict["kernel_func_name"] + ".cu"
                            )
                            write_kernel_to_file(kernel_str, file_name)

                            all_kernel_names += (
                                kernel_dict["kernel_func_name"] + ", \n"
                            )
                            all_kernel_declares += (
                                "cutlass::Status "
                                + kernel_dict["kernel_func_name"]
                                + "(const FcAllParams& params);"
                                + "\n"
                            )

        # Generate op code
        op_dict["kernel_func_declare"] = all_kernel_declares
        op_dict["all_kernel_func_name"] = all_kernel_names
        sm75_code += SubstituteTemplate(CommonFcFunction, op_dict)
    return sm75_code     
    
    
def sm80_16816_forStreamK(op_dict, kernel_dict, suffix, epi_func):
    all_kernel_names = ""
    all_kernel_declares = ""
    kernel_dict["swizzling_functor"] = 'ThreadblockSwizzleStreamK'
    kernel_dict["split_k_factor"] = "1"
    kernel_dict["kernel_func_name"] = op_dict["func_name"] + "_" + str(suffix)
    suffix += 1
    
    fba_kernel_ = fba_kernel
    if epi_func in [FbaAct.LeakyRelu]:
        fba_kernel_ = fba_kernel_leaky_alpha
    kernel_str = (
        fba_header + SubstituteTemplate(fba_kernel_, kernel_dict) + CommonTail
    )
    file_name = (
        build_dir + "generated_tmp/" + kernel_dict["kernel_func_name"] + ".cu"
    )
    write_kernel_to_file(kernel_str, file_name)

    all_kernel_names += (
        kernel_dict["kernel_func_name"] + ", \n"
    )
    all_kernel_declares += (
        "cutlass::Status "
        + kernel_dict["kernel_func_name"]
        + "(const FcAllParams& params);"
        + "\n"
    )
    return all_kernel_names, all_kernel_declares, suffix


def sm80_16816_forUniversal(op_dict, kernel_dict, suffix, epi_func):
    all_kernel_names = ""
    all_kernel_declares = ""
    kernel_dict["swizzling_functor"] = 'GemmIdentityThreadblockSwizzle<1>'
    for split_k_fac in split_k_factors:
        kernel_dict["split_k_factor"] = split_k_fac
        kernel_dict["kernel_func_name"] = op_dict["func_name"] + "_" + str(suffix)
        suffix += 1
        
        fba_kernel_ = fba_kernel
        if epi_func in [FbaAct.LeakyRelu]:
            fba_kernel_ = fba_kernel_leaky_alpha
        kernel_str = (
            fba_header + SubstituteTemplate(fba_kernel_, kernel_dict) + CommonTail
        )
        file_name = (
            build_dir + "generated_tmp/" + kernel_dict["kernel_func_name"] + ".cu"
        )
        write_kernel_to_file(kernel_str, file_name)

        all_kernel_names += (
            kernel_dict["kernel_func_name"] + ", \n"
        )
        all_kernel_declares += (
            "cutlass::Status "
            + kernel_dict["kernel_func_name"]
            + "(const FcAllParams& params);"
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
    # 以下三个和alignments需要对齐一下
    kernel_dict["epilogue_vector_length"] = "8"
    kernel_dict["align_a"] = "8"
    kernel_dict["align_b"] = "8"

    math_instructions = [
        ("16,8,16", cutlass_dtype, cutlass_dtype, "float")
    ]

    sm80_code = ""
    for epi_func in SupportedAct:
        op_dict = {}
        op_dict["func_name"] = (
            UnderScoreName[epi_func].lower() + "_sm80_"
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
                        # TileDesc("256, 128, 32", 3, "64, 64, 32", math_inst),
                        # TileDesc("128, 256, 32", 3, "64, 64, 32", math_inst),
                        # TileDesc("256, 64, 32", 3, "64, 64, 32", math_inst),
                        # TileDesc("256, 64, 32", 4, "64, 64, 32", math_inst),
                        # TileDesc("64, 256, 32", 4, "64, 64, 32", math_inst),
                        # TileDesc("128, 128, 32", 3, "64, 64, 32", math_inst),
                        # TileDesc("128, 128, 32", 4, "64, 64, 32", math_inst),
                        # TileDesc("128, 128, 32", 5, "64, 64, 32", math_inst),
                        # TileDesc("128, 64, 32", 6, "64, 32, 32", math_inst),
                        # TileDesc("64, 128, 32", 6, "32, 64, 32", math_inst),
                        # TileDesc("64, 64, 32", 10, "32, 32, 32", math_inst),
                        # TileDesc("256, 128, 64", 3, "64, 64, 64", math_inst),
                        # TileDesc("128, 256, 64", 3, "64, 64, 64", math_inst),
                        # TileDesc("256, 64, 64", 4, "64, 64, 64", math_inst),
                        # TileDesc("64, 256, 64", 4, "64, 64, 64", math_inst),
                        # TileDesc("128, 128, 64", 4, "64, 64, 64", math_inst),
                        # TileDesc("256, 64, 64", 3, "64, 64, 64", math_inst),
                        # TileDesc("64, 256, 64", 3, "64, 64, 64", math_inst),
                        # TileDesc("128, 128, 64", 3, "64, 64, 64", math_inst),
                        # TileDesc("128, 64, 64", 3, "64, 32, 64", math_inst),
                        # TileDesc("64, 128, 64", 3, "32, 64, 64", math_inst),
                        # TileDesc("64, 64, 64", 5, "32, 32, 64", math_inst),
                        # kai mod
                        TileDesc("16, 64, 64", 5, "16, 32, 64", math_inst),
                    ]
                    for tile in tiles:
                        kernel_dict["Tshape"] = tile.Tshape
                        kernel_dict["Wshape"] = tile.Wshape
                        kernel_dict["Ishape"] = tile.math_inst[0]
                        kernel_dict["stages"] = str(tile.stages)
                        kernel_dict["element_accum"] = tile.math_inst[3]
                        all_kernel_names_universal, all_kernel_declares_universal, suffix = sm80_16816_forUniversal(op_dict, kernel_dict, suffix, epi_func)
                        all_kernel_names_streamk, all_kernel_declares_streamk, suffix = sm80_16816_forStreamK(op_dict, kernel_dict, suffix, epi_func)
                        all_kernel_declares += all_kernel_declares_universal
                        all_kernel_declares += all_kernel_declares_streamk
                        all_kernel_names += all_kernel_names_universal  
                        all_kernel_names += all_kernel_names_streamk     
        # Generate op code
        op_dict["kernel_func_declare"] = all_kernel_declares
        op_dict["all_kernel_func_name"] = all_kernel_names
        sm80_code += SubstituteTemplate(CommonFcFunction, op_dict)
    return sm80_code                


if __name__ == "__main__":
    sm_versions_and_types = []
    args = parse_args()
    all_code = mutex_include + fba_header
    # 暂时没有考虑fp32
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
    with open(build_dir + "generated_tmp/fc_bias_act.cu", "w") as f:
        f.write(all_code)
        f.close()