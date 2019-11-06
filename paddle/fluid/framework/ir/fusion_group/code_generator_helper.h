/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace paddle {
namespace framework {
namespace ir {
namespace fusion_group {

static std::vector<std::string> replaced_element_in_order = {"@", "$"};

static std::vector<std::string> kernel_template = {"$func_name", "$parameters",
                                                   "$compute_body"};

class OperationExpression {
 public:
  explicit OperationExpression(std::string op, std::vector<int> input_ids,
                               std::vector<int> output_ids)
      : op_(op), input_ids_(input_ids), output_ids_(output_ids) {}

  std::vector<int> GetInputIds() { return input_ids_; }
  std::vector<int> GetOutputIds() { return output_ids_; }

  // Check whether this operation type is supported in OperationMap.
  bool IsSupport();

  std::string GetExpression();

  // TODO(wangchao): make offset more flexible we add stride and basic offset
  std::string GetRHSTemplate(size_t i = 0);
  std::string GetLHSTemplate(size_t i = 0);

 private:
  std::string op_;
  std::vector<int> input_ids_;
  std::vector<int> output_ids_;
};

class TemplateVariable {
 public:
  void Add(std::string identifier, std::string expression) {
    strings_[identifier] = expression;
  }
  void Remove(std::string identifier, std::string expression) {
    for (auto it = strings_.begin(); it != strings_.end();) {
      if (it->first == identifier) {
        it = strings_.erase(it);
      } else {
        it++;
      }
    }
  }

  std::unordered_map<std::string, std::string> Get() { return strings_; }

 private:
  std::unordered_map<std::string, std::string> strings_;
};

class CodeTemplate {
 public:
  CodeTemplate() = default;
  explicit CodeTemplate(std::string template_str) {
    template_str_ = template_str;
  }

  std::string Format(TemplateVariable template_var) {
    std::string ret = template_str_;
    std::unordered_map<std::string, std::string> identifier_str =
        template_var.Get();

    for (size_t i = 0; i < ret.size(); i++) {
      auto pos = i;
      char c = ret[pos];

      if (c == '$') {
        for (size_t j = 0; j < kernel_template.size(); j++) {
          int template_size = kernel_template[j].size();
          auto tmp_cmp = ret.substr(pos, template_size);
          if (tmp_cmp == kernel_template[j]) {
            ret.replace(pos, template_size, identifier_str[kernel_template[j]]);
          }
        }
      }
    }

    return EmitIndents(ret);
  }

  std::string EmitIndents(std::string str) {
    std::string ret = str;
    int space_num = 0;
    auto space_char = ' ';
    for (size_t i = 0; i < ret.size(); i++) {
      auto pos = i;
      char c = ret[pos];
      if (c == '\n') {
        size_t next_pos = pos + 1;
        while (next_pos < ret.size() && ret[next_pos] == space_char) {
          next_pos++;
        }
        space_num = next_pos - pos - 1;
      }
      if (c == ';' && (pos + 1 < ret.size()) && ret[pos + 1] != '\n') {
        auto insert_pos = pos + 1;
        std::string insert_str = "\n" + std::string(space_num, space_char);
        ret.insert(insert_pos, insert_str);
        space_num = 0;
      }
    }

    return ret;
  }

 private:
  std::string template_str_;
};

static const char predefined_cuda_functions[] = R"(
__device__ float real_exp(float x) { return ::expf(x); }
__device__ double real_exp(double x) { return ::exp(x); }

__device__ float real_log(float x) { return ::logf(x); }
__device__ double real_log(double x) { return ::log(x); }

__device__ float real_min(float x, float y) { return ::fminf(x, y); }
__device__ double real_min(double x, double y) { return ::fmin(x, y); }

__device__ float real_max(float x, float y) { return ::fmaxf(x, y); }
__device__ double real_max(double x, double y) { return ::fmax(x, y); }

)";

static const char elementwise_cuda_template[] = R"(

extern "C" __global__ void $func_name($parameters) {
  for(int idx = blockIdx.x * blockDim.x + threadIdx.x;
      idx < N;
      idx += gridDim.x * blockDim.x) {
    $compute_body
  }
}
)";

}  // namespace fusion_group
}  // namespace ir
}  // namespace framework
}  // namespace paddle
