/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#pragma once

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace paddle {
namespace framework {
namespace ir {
static std::unordered_map<std::string, std::string> operator_cuda_table = {
    {"elementwise_add", "var$ + var$"},
    {"elementwise_sub", "var$ - var$"},
    {"elementwise_mul", "var$ * var$"},
    {"elementwise_div", "var$ / var$"},
    {"elementwise_min", "real_min(var$, var$)"},
    {"elementwise_max", "real_max(var$, var$)"},
    {"relu", "real_max(var$, 0)"},
    {"sigmoid", "1.0 / (1.0 + real_exp(-var$))"}};

// op computation is composed by single or many operation
class OperationExpression {
 public:
  OperationExpression(std::vector<int> input_ids, int output_id,
                      std::string search_oprtation);
  std::string GetExpression();
  std::vector<int> GetInputIds() { return input_ids_; }
  int GetOutputId() { return output_id_; }

 private:
  std::vector<int> input_ids_;
  int output_id_;
  std::string search_operation_;
};

static const char indentation[] = R"(    )";

static const char const_kernel_start[] = R"(
template <typename T>
extern "C" __global__ void
)";

static const char const_kernel_mid[] = R"(
{
  for(int idx = blockIdx.x * blockDim.x + threadIdx.x;
      idx < N;
      idx += gridDim.x * blockDim.x) {

)";

static const char const_kernel_end[] = R"(
}
}
)";
}  // namespace ir
}  // namespace framework
}  // namespace paddle
