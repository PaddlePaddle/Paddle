# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys

dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))
sys.path.append(dirname + "/../")
from util import SubstituteTemplate

# For beginners, these template parameters may be difficult to understand.
# Please refer to the conv-related demo of CUTLASS for better understanding.
# https://github.com/NVIDIA/cutlass/tree/master/examples

CommonCutlassConvKernelDeclare = """
cutlass::Status ${kernel_func_name}(const ConvAllParams& params) {
  using kernel_base =
  typename cutlass::conv::kernel::${conv_kind_name}<
    ${element_a},
    ${layout_a},
    ${element_b},
    ${layout_b},
    ${element_c},
    ${layout_c},
    ${element_accum},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${Tshape}>,
    cutlass::gemm::GemmShape<${Wshape}>,
    cutlass::gemm::GemmShape<${Ishape}>,
    ${epi_part},
    ${swizzling_functor},
    ${stages},
    ${math_operator},
    ${iterator_algorithm},
    ${stride_support},
    ${align_a},
    ${align_b}
  >::Kernel;

  using ImplicitGemm =
      cutlass::conv::device::ImplicitGemmConvolution<kernel_base>;

  ${element_a} *input = (${element_a} *)(params.input);
  ${element_b} *weight = (${element_b} *)(params.weight);
  ${element_c} *bias = (${element_c} *)(params.bias);
  ${element_c} *output = (${element_c} *)(params.output);
  // only used by conv2d_bias_residual
  auto residual = (${element_c} *)(params.residual);

  int batch = params.batch;
  int ic = params.ic;
  int ih = params.ih;
  int iw = params.iw;
  int kh = params.kh;
  int kw = params.kw;
  int oc = params.oc;
  int pad_h0 = params.pad_h0;
  int pad_w0 = params.pad_w0;
  int stride_h = params.stride_h;
  int stride_w = params.stride_w;
  int groups = params.groups;
  int kc = ic / groups;

  int oh = params.oh;
  int ow = params.ow;
  int dilation_h = params.dilation_h;
  int dilation_w = params.dilation_w;
  int split_k_slices = ${split_k_slices};

  cutlass::conv::Conv2dProblemSize problem_size({batch, ih, iw, ic},
                                                {oc, kh, kw, ic / groups},
                                                {pad_h0, 0, pad_w0, 0},
                                                {stride_h, stride_w},
                                                {dilation_h, dilation_w},
                                                {batch, oh, ow, oc},
                                                cutlass::conv::Mode::kCrossCorrelation,
                                                split_k_slices,
                                                groups);
"""

# This is the execution part of this cutlass conv kernel.

CommonCutlassConvKernelExecute = """
  ImplicitGemm implicit_gemm_op;
  size_t bytes = implicit_gemm_op.get_workspace_size(arguments);

  auto stream = params.stream;
  void *workspace = params.workspace;

  cutlass::Status status = implicit_gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = implicit_gemm_op.initialize(arguments, workspace);
  CUTLASS_CHECK(status);
  status = implicit_gemm_op(stream);
  CUTLASS_CHECK(status);
  return status;
}
"""

# CommonConvFunction is a wrapper for many kernels
# a func_name is like conv2d_bias_silu_sm75
# it has many kernels, we should pick up a performance-best
# ${func_name} is like conv2d_bias_silu_sm75
# ${enum_op_name} is like CONV2D_BIAS_SILU

CommonConvFunction = """

${kernel_func_declare}

std::vector<std::function<cutlass::Status(const ConvAllParams)>>
    ${func_name}_all_func =  {${all_kernel_func_name}};

std::map<std::vector<int>, int> map_problem_${func_name};
std::mutex ${func_name}_mutex;

bool ${func_name}(ConvAllParams params) {
  int batch = params.batch;
  int ic = params.ic;
  int ih = params.ih;
  int iw = params.iw;
  int kh = params.kh;
  int kw = params.kw;
  int oc = params.oc;
  //int pad_h0 = params.pad_h0;
  //int pad_w0 = params.pad_w0;
  int groups = params.groups;
  int stride_h = params.stride_h;
  int stride_w = params.stride_w;

  std::vector<int> problem_size = {
      batch, ic, ih, iw, kh, kw, oc, groups, stride_h, stride_w};

  if (map_problem_${func_name}.count(problem_size)) {
    ${func_name}_all_func[map_problem_${func_name}.at(problem_size)](
        params);
    return true;
  }

  int best_config_index = ProfileToGetBestConfig(
      ${func_name}_all_func, params, ${enum_op_name});

  std::lock_guard<std::mutex> guard(${func_name}_mutex);

  map_problem_${func_name}[problem_size] = best_config_index;
  ${func_name}_all_func[best_config_index](params);
  return true;
}
"""


# We should wrapper all op_name_with_sm_version into a function
# like : wrapper conv2d_bias_silu_sm75,  conv2d_bias_silu_sm80,  conv2d_bias_silu_sm86 into conv2d_bias_silu for phi kernel
# this function is invoked by phi kernel

CommonWrapperForPhi = """
bool ${op_name}(ConvAllParams params) {
  ${dispatch_body}
}
"""


def convert_c_data_type(dtype):
    if dtype == "fp16":
        return "Conv2dDataType::fp16"
    elif dtype == "bf16":
        return "Conv2dDataType::bf16"
    elif dtype == "fp32":
        return "Conv2dDataType::fp32"
    else:
        return None


CommonDispatchTemp = '''
    if (params.sm_version == ${sm_code} && params.data_type == ${data_type})
    {
        return ${op_name_with_sm}(params);
    }
    '''


# this is a file's ending part

CommonTail = '''
}  // namespace cutlass_internal
}  // namespace fusion
}  // namespace phi
'''


# Wrap different sm versions into a function called by phi
def GenerateFunctionForPhi(
    sm_versions_and_types, support_epi_funcs, underscore_names, camel_names
):
    generated_code = ""
    for epi_func in support_epi_funcs:
        dispatch_body = ""
        for sm_version, data_type in sm_versions_and_types:
            sm_dicts = {}
            sm_dicts["sm_code"] = sm_version
            sm_dicts["data_type"] = convert_c_data_type(data_type)
            sm_dicts["op_name_with_sm"] = (
                underscore_names[epi_func].lower()
                + "_sm"
                + sm_version
                + "_"
                + data_type
            )
            dispatch_body += SubstituteTemplate(CommonDispatchTemp, sm_dicts)
        dispatch_body += '''    return false;'''
        op_dicts = {}
        op_dicts["dispatch_body"] = dispatch_body
        op_dicts["op_name"] = camel_names[epi_func]
        generated_code += SubstituteTemplate(CommonWrapperForPhi, op_dicts)
    return generated_code


# We modify some template parameters based on CommonCutlassConvKernelDeclare.
CommonCutlassConv2dDepthwiseKernelDeclare = (
    CommonCutlassConvKernelDeclare.replace(
        "${align_a}", "cutlass::MatrixShape<${strided_shape}>"
    )
    .replace("${align_b}", "cutlass::MatrixShape<${dilation_shape}>")
    .replace("ImplicitGemmConvolution", "DirectConvolution")
    .replace(
        "cutlass::gemm::GemmShape<${Tshape}>,",
        '''cutlass::gemm::GemmShape<${Tshape}>,
       cutlass::conv::TensorNHWCShape<${T_output_shape}>,
       cutlass::MatrixShape<${filter_shape}>,
     ''',
    )
)
