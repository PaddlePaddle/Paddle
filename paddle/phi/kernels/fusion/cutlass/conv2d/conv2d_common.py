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

# a function name is like such as conv2d_bias_silu_sm75
# it has many kernels, we should pick up a performence-best
# ${func_name} is like conv2d_bias_silu_sm75
# ${enum_op_name} is like CONV2D_BIAS_SILU

common_conv_function = """
std::vector<std::function<cutlass::Status(ConvAllParams)>>
    ${func_name}_all_func =  {${all_kernel_func_name}};

std::map<std::vector<int>, int> map_problem_${func_name};
std::mutex ${func_name}_mutex;

void ${func_name}(ConvAllParams params) {
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

  std::vector<int> problem_size = {
      batch, ic, ih, iw, kh, kw, oc, pad_h0, pad_w0, stride_h, stride_w};

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
"""


# We should wrapper all op_name_with_sm_version into a function
# like : wrapper conv2d_bias_silu_sm75,  conv2d_bias_silu_sm80,  conv2d_bias_silu_sm86 into conv2d_bias_silu for phi kernel
# this function is invoked by phi kernel

common_wrapper_for_phi = """
void ${op_name}(ConvAllParams params) {
    ${dispatch_body}
}
"""


common_dispatch_temp = '''
    if (params.sm_version == ${sm_code})
    {
        ${op_name_with_sm}(params);
    }
    '''


# this is a file's ending part

common_tail = '''
}  // namespace cutlass_internal
}  // namespace fusion
}  // namespace phi
'''
