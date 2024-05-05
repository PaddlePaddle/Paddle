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

import re
from argparse import ArgumentParser


def SubstituteTemplate(template, values):
    text = template
    changed = True
    while changed:
        changed = False
        for key, value in values.items():
            regex = "\\$\\{%s\\}" % key
            newtext = re.sub(regex, value, text)
            if newtext != text:
                changed = True
            text = newtext
    return text


code_template = """
#include "paddle/extension.h"
#include "${triton_kernel_header_file}"


std::map<std::vector<int>, int> map_problem_${custom_op_name}_func;


void ${custom_op_name}_func(${para}) {


auto get_tensor_ptr = [](const paddle::Tensor& input) -> CUdeviceptr {
  if (input.type() == paddle::DataType::FLOAT16) {
    return (CUdeviceptr)(input.data<phi::dtype::float16>());
  } else if (input.type() == paddle::DataType::INT32) {
    return (CUdeviceptr)(input.data<int>());
  } else if (input.type() == paddle::DataType::FLOAT32) {
    return (CUdeviceptr)(input.data<float>());
  } else if (input.type() == paddle::DataType::UINT8) {
    return (CUdeviceptr)(input.data<uint8_t>());
  } else if (input.type() == paddle::DataType::INT8) {
    return (CUdeviceptr)(input.data<int8_t>());
  } else {
    assert(false);
    return (CUdeviceptr)(nullptr);
  }
};

  std::vector<int> problem_size = {${key}};

  if (map_problem_${custom_op_name}_func.count(problem_size)) {
    int algo_id = map_problem_${custom_op_name}_func[problem_size];
    auto status = ${triton_kernel}(${invoke_para} algo_id);
    assert(status == CUDA_SUCCESS);
    return;
  }
  std::cout << "we are tuning for ${custom_op_name} which key is: ";
  for (int i = 0; i < problem_size.size(); i++) {
    std::cout << problem_size[i] << ", ";
  }

  float min_time = 10000.f;
  int select_id = -1;
  constexpr int WARMUP = 5;
  constexpr int REPEAT = 10;

  for (int algo_id = 0; algo_id < ${triton_kernel}_get_num_algos(); ++algo_id) {
    cudaEvent_t beg[REPEAT];
    cudaEvent_t end[REPEAT];
    float elapsed_times[REPEAT];

    auto status = CUDA_SUCCESS;

    for (int ii = 0; ii < WARMUP + REPEAT; ii++) {
      int repeat_id = ii - WARMUP;

      if (repeat_id >= 0) {
        (cudaEventCreate(beg + repeat_id));
        (cudaEventCreate(end + repeat_id));
        (cudaEventRecord(beg[repeat_id]));
      }

      //auto flush_l2_cache = paddle::full({10 * 1024 * 1024}, 0, paddle::DataType::INT32, param0.place());
      // std::cout << &flush_l2_cache  << std::endl;

      status = ${triton_kernel}(${invoke_para} algo_id);

      if (repeat_id >= 0) {
        (cudaEventRecord(end[repeat_id]));
        (cudaEventSynchronize(end[repeat_id]));
        (cudaEventElapsedTime(
            elapsed_times + repeat_id, beg[repeat_id], end[repeat_id]));
      }
    }

    float avg_elapsed_time = 0.f;
    for (int ii = 0; ii < REPEAT; ++ii) {
      avg_elapsed_time += elapsed_times[ii];
    }

    if (avg_elapsed_time < min_time && status == CUDA_SUCCESS) {
      min_time = avg_elapsed_time;
      select_id = algo_id;
    }
  }
  assert(select_id >= 0);
  map_problem_${custom_op_name}_func[problem_size] = select_id;
  std::cout << "select algo id: " << select_id << std::endl;
}

PD_BUILD_OP(${custom_op_name})
    .Inputs({${op_inputs}})
    .Outputs({${op_outputs}})
    .SetInplaceMap({${inplace_map}})
    .Attrs({${op_attrs}})
    .SetKernelFn(PD_KERNEL(${custom_op_name}_func));
"""


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--custom-op-name",
        "-on",
        type=str,
        default="",
        help="Name of the paddle custom op",
        required=True,
    )
    parser.add_argument(
        "--custom-op-file",
        "-cof",
        type=str,
        help="The file.cu of generated .cu file",
        required=True,
    )
    parser.add_argument(
        "--kernel-name",
        "-kn",
        type=str,
        default="",
        help="Name of the triton kernel you will invoke",
        required=True,
    )
    parser.add_argument(
        "--signature",
        "-s",
        type=str,
        help="Signature of the triton kernel",
        required=True,
    )
    parser.add_argument(
        "--output-ids",
        "-oi",
        type=str,
        help="Output ids in your signature",
        required=True,
    )
    parser.add_argument(
        "--triton-kernel-header-file",
        "-header",
        type=str,
        help="The header file of the triton kernel",
        required=True,
    )
    parser.add_argument(
        "--key-attr-id",
        "-kid",
        type=str,
        help="The key of this kernel",
        required=True,
    )

    args = parser.parse_args()

    template_dict = {}
    template_dict["triton_kernel"] = args.kernel_name
    template_dict["custom_op_name"] = args.custom_op_name
    signature = args.signature
    signature = signature.split(",")

    def convert_to_ctype(sig):
        if "i32" in sig:
            return "int"
        elif "fp32" in sig:
            return "float"

    op_inputs_len = 0
    attrs_type = []

    for i in range(len(signature)):
        sig = signature[i]
        sig = sig.strip(" ")
        if "*" in sig:
            op_inputs_len += 1
        elif sig.isdigit():
            print("this is a mata-parameter")
            pass
        else:
            attrs_type.append(convert_to_ctype(sig))

    para = ""
    for i in range(op_inputs_len):
        para += f"const paddle::Tensor& para{i},"
    for i in range(len(attrs_type)):
        para += attrs_type[i] + f" attr{i},"
    para = para[:-1]

    template_dict["para"] = para

    invoke_para = "para0.stream(),"
    for i in range(op_inputs_len):
        invoke_para += f"get_tensor_ptr(para{i}),"
    for i in range(len(attrs_type)):
        invoke_para += f"attr{i},"
    template_dict["invoke_para"] = invoke_para

    op_inputs = ""
    for i in range(op_inputs_len):
        op_inputs += f"\"para{i}\", "
    op_inputs = op_inputs[:-2]
    template_dict["op_inputs"] = op_inputs

    op_attrs = ""
    for i in range(len(attrs_type)):
        op_attrs += f"\"attr{i} : {attrs_type[i]}\", "
    # remove the last ","
    op_attrs = op_attrs[:-1]

    template_dict["op_attrs"] = op_attrs

    output_ids = args.output_ids
    output_ids = output_ids.split(",")
    op_outputs = ""
    for id in output_ids:
        op_outputs += f"\"out{id}\","
    op_outputs = op_outputs[:-1]
    template_dict["op_outputs"] = op_outputs

    inplace_map = ""
    for id in output_ids:
        inplace_map += f"{{\"para{id}\",\"out{id}\"}},"
    inplace_map = inplace_map[:-1]

    template_dict["inplace_map"] = inplace_map
    template_dict["triton_kernel_header_file"] = args.triton_kernel_header_file

    key_ids = args.key_attr_id.split(",")
    key = ""
    for i in key_ids:
        key += f"attr{i},"
    key = key[:-1]
    template_dict["key"] = key

    file_path = args.custom_op_file
    with open(file_path, "w") as f:
        f.write(SubstituteTemplate(code_template, template_dict))
        f.close()
