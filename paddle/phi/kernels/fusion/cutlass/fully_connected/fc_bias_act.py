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

// namespace phi{
// namespace fusion{
// namespace cutlass_internal{
'''


# cutlass::epilogue::thread::LinearCombinationRelu<cutlass::half_t, 8, float, float>
dict_for_declare_part = {
    "swizzling_functor": "cutlass::gemm::threadblock::${threadblock_swizzle}",
    "epi_part": "${epi_func}< ${element_c}, ${epilogue_vector_length}, ${element_accum}, ${element_epilogue}, ${scale_type}>",
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

SupportedAct = [
    FbaAct.Identity,
    FbaAct.Relu,
    FbaAct.Silu,
    FbaAct.LeakyRelu,
    FbaAct.Sigmoid,
]

ActTag = {
    SupportedAct[0]: 'cutlass::epilogue::thread::LinearCombination',
    SupportedAct[1]: 'cutlass::epilogue::thread::LinearCombinationRelu',
    SupportedAct[2]: 'cutlass::epilogue::thread::LinearCombinationSilu',
    SupportedAct[3]: 'cutlass::epilogue::thread::LinearCombinationLeakyRelu',
    SupportedAct[4]: 'cutlass::epilogue::thread::LinearCombinationSigmoid',
}

UnderScoreName = {
    SupportedAct[0]: "fc_bias",
    SupportedAct[1]: "fc_bias_relu",
    SupportedAct[2]: "fc_bias_silu",
    SupportedAct[3]: "fc_bias_leaky_relu",
    SupportedAct[4]: "fc_bias_sigmoid",
}

CamelName = {
    SupportedAct[0]: "FcBias",
    SupportedAct[1]: "FcBiasRelu",
    SupportedAct[2]: "FcBiasSilu",
    SupportedAct[3]: "FcBiasLeakyRelu",
    SupportedAct[4]: "FcBiasSigmoid",
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

swizzling_functor = [
    'GemmIdentityThreadblockSwizzle<1>',         # cutlass 写死是1的 我理解就等同于没有swizzle 或许是它通用的原因？
    'ThreadblockSwizzleStreamK'
]

# 暂时没有考虑fp32 所以只用128bit对齐 128/sizeof(half)
alignments = [8]

def generate_sm75_1688():
    print("generate_sm75_1688")

def generate_sm80_16816(cutlass_dtype="cutlass::half_t"):
    kernel_dict = {
        "element_a": cutlass_dtype,
        "element_b": cutlass_dtype,
        "element_c": cutlass_dtype,
        "opcode_class": "cutlass::arch::OpClassTensorOp",
        "arch": "cutlass::arch::Sm80",
        # TODO 这个参数待修改
        "threadblock_swizzle": swizzling_functor[0],
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

    # TODO 这些参数待修改
    kernel_dict["split_k_factor"] = "1"
    kernel_dict["avail_sms"] = ""
    
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
            for layout_iter in range(len(layouts)):
                kernel_dict["layout_a"] = layouts[layout_iter][0]
                kernel_dict["layout_b"] = layouts[layout_iter][1]
                kernel_dict["layout_c"] = layouts[layout_iter][2]
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
                        TileDesc("64, 64, 64", 5, "32, 32, 64", math_inst)]
                    for tile in tiles:
                        kernel_dict["Tshape"] = tile.Tshape
                        kernel_dict["Wshape"] = tile.Wshape
                        kernel_dict["Ishape"] = tile.math_inst[0]
                        kernel_dict["stages"] = str(tile.stages)
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