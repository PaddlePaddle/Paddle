import os
import sys

dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))
sys.path.append(dirname + "/../")
from util import SubstituteTemplate

# 对于GemmUniversal C和D的数据类型是一样的 Accumulator固定为float Epilogue固定为float
CommonCutlassFcKernelDeclare = '''
cutlass::Status ${kernel_func_name}(const FcAllParams& params) {
    /// CommonCutlassFcKernelDeclare
    using DeviceKernalName = cutlass::gemm::device::GemmUniversal<
        ${element_a}, ${layout_a},
        ${element_b}, ${layout_b},
        ${element_c}, ${layout_c},
        ${element_accum},
        ${opcode_class},
        ${arch},
        cutlass::gemm::GemmShape<${Tshape}>,
        cutlass::gemm::GemmShape<${Wshape}>,
        cutlass::gemm::GemmShape<${Ishape}>,
        ${epi_part},
        ${swizzling_functor},
        ${stages},          // num_stage
        ${align_a},         // AlignA
        ${align_b},         // AlignB
        ${math_operator}    // Operation performed by GEMM
    >;

'''

CommonCutlassFcKernelArguments = '''
    /// Arguments
    cutlass::gemm::GemmCoord problem_size{params.m, params.n, params.k};
    ${element_a} *input = (${element_a} *)(params.input);
    ${element_b} *weight = (${element_b} *)(params.weight);
    ${element_c} *bias = (${element_c} *)(params.bias);
    ${element_c} *output = (${element_c} *)(params.output);
    // 这些都是layout的Demo写法 实际上是如何安排这些layout组合的计算的 我还不清楚
    int64_t batch_stride_C = problem_size.n();
    long lda = (long)params.lda;   
    long ldb = (long)params.ldb;
    long ldc_bias = 0;
    long ldd = (long)params.ldd;

    typename DeviceKernalName::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_size,
        ${split_k_factor},
        {1.f, 1.f},
        input,
        weight,
        bias,
        output,
        problem_size.mk().product(),
        problem_size.nk().product(),
        batch_stride_C,
        problem_size.mn().product(),
        lda,
        ldb,
        ldc_bias,
        ldd,
        ${avail_sms}
    };
'''

CommonCutlassFcKernelExecute = '''

    /// CommonCutlassFcKernelExecute
    DeviceKernalName device_gemm;
    size_t workspace_size = DeviceKernalName::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = device_gemm.can_implement(arguments);
    CUTLASS_CHECK(status);
    status = device_gemm.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);
    status = device_gemm();
    CUTLASS_CHECK(status);
    return status;
}
'''


CommonFcFunction = '''
${kernel_func_declare}

std::vector<std::function<cutlass::Status(const FcAllParams)>>
    ${func_name}_all_func =  {${all_kernel_func_name}};

std::map<std::vector<int>, int> map_problem_${func_name};
std::mutex ${func_name}_mutex;

void ${func_name}(FcAllParams params) {
  int m = params.m;
  int n = params.n;
  int k = params.k;
  int lda = params.lda;
  int ldb = params.ldb;
  int ldd = params.ldd;

  std::vector<int> problem_size = {m, n, k, lda, ldb, ldd};

  if (map_problem_${func_name}.count(problem_size)) {
    ${func_name}_all_func[map_problem_${func_name}.at(problem_size)](
        params);
    return;
  }

  int best_config_index = ProfileToGetBestConfig(
      ${func_name}_all_func, params, ${enum_op_name});

  std::lock_guard<std::mutex> guard(${func_name}_mutex);

  map_problem_${func_name}[problem_size] = best_config_index;
  ${func_name}_all_func[best_config_index](params);
}
'''



CommonTail = '''

// }  // namespace cutlass_internal
// }  // namespace fusion
// }  // namespace phi
'''



CommonWrapperForPhi = """
void ${op_name}(FcAllParams params) {
    ${dispatch_body}
}
"""

CommonDispatchTemp = '''
    if (params.sm_version == ${sm_code} && params.data_type == ${data_type})
    {
        ${op_name_with_sm}(params);
    }
'''

def convert_c_data_type(dtype):
    if dtype == "fp16":
        return "FcDataType::fp16"
    if dtype == "bf16":
        return "FcDataType::bf16"


def GenerateFunctionForPhi(
    sm_versions_and_types, support_epi_funcs, underscore_names, camel_names):
    generated_code = ""
    for epi_fun in support_epi_funcs:
        dispatch_body = ""
        for sm_version, data_type in sm_versions_and_types:
            sm_dicts = {}
            sm_dicts["sm_code"] = sm_version
            sm_dicts["data_type"] = convert_c_data_type(data_type)
            sm_dicts["op_name_with_sm"] = (
                underscore_names[epi_fun].lower() + "_sm" + sm_version + "_" + data_type
            )
            dispatch_body += SubstituteTemplate(CommonDispatchTemp, sm_dicts)
        op_dicts = {}
        op_dicts["dispatch_body"] = dispatch_body
        op_dicts["op_name"] = camel_names[epi_fun]
        generated_code += SubstituteTemplate(CommonWrapperForPhi, op_dicts)
    return generated_code