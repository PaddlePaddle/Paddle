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

static std::vector<std::string> replaced_element_in_order = {"@", "$"};

static std::vector<std::string> kernel_template = {"$name", "$parameter",
                                                   "$compute"};

static std::unordered_map<std::string, std::string> support_table = {
    {"elementwise_add", "var@ + var$"},
    {"elementwise_sub", "var@ - var$"},
    {"elementwise_mul", "var@ * var$"},
    {"elementwise_div", "var@ / var$"},
    {"elementwise_min", "real_min(var@, var$)"},
    {"elementwise_max", "real_max(var@, var$)"},
    {"relu", "real_max(var@, 0)"},
    {"sigmoid", "1.0 / (1.0 + real_exp(-var@))"}};

// Paddle elementwise op consist the broacast op and elementwise op
// op computation is composed by single or many operation
// here we only generate the simple expression code so we
// make it simple
class OperationExpression {
 public:
  OperationExpression(std::vector<int> input_ids, int output_id,
                      std::string op);
  std::string GetExpression();
  std::vector<int> GetInputIds() { return input_ids_; }
  int GetOutputId() { return output_id_; }
  bool SupportState();
  // in oreder to make offset more flexible we add stride and basic offset
  std::string GetRHSTemplate();
  std::string GetLHSTemplate();

 private:
  std::vector<int> input_ids_;
  int output_id_;
  std::string op_;
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

static std::string EmitUniqueName(std::vector<OperationExpression> expression) {
  std::stringstream ret;
  ret << "fused_kernel";
  for (size_t i = 0; i < expression.size(); i++) {
    ret << expression[i].GetOutputId();
  }
  return ret.str();
}
// we get the parameter list code for the expression information
static std::string EmitDeclarationCode(
    std::vector<OperationExpression> expression, std::string type) {
  std::stringstream ret;

  std::set<int> input_ids;
  std::set<int> output_ids;

  for (size_t i = 0; i < expression.size(); i++) {
    std::vector<int> tmp_input = expression[i].GetInputIds();
    for (size_t j = 0; j < tmp_input.size(); j++) {
      int id = tmp_input[j];
      input_ids.insert(id);
    }
    int tmp_output = expression[i].GetOutputId();
    output_ids.insert(tmp_output);
  }

  std::set<int>::iterator it = input_ids.begin();
  while (it != input_ids.end()) {
    int var_index = *it;
    if (output_ids.find(var_index) != output_ids.end()) {
      input_ids.erase(it++);
    } else {
      it++;
    }
  }

  ret << "int N, ";
  for (it = input_ids.begin(); it != input_ids.end(); it++) {
    int var_index = *it;
    ret << type << R"(* var)" << var_index;
    ret << ", ";
  }

  size_t count_index = 0;
  for (it = output_ids.begin(); it != output_ids.end(); it++) {
    int var_index = *it;
    ret << type << R"(* var)" << var_index;
    if (count_index != output_ids.size() - 1) {
      ret << ", ";
    }
    count_index++;
  }

  return ret.str();
}

static std::string EmitComputeCode(
    std::vector<OperationExpression> expression) {
  // get the right experssion code using suffix expression
  std::stringstream ret;
  for (size_t i = 0; i < expression.size(); i++) {
    ret << expression[i].GetExpression();
  }
  return ret.str();
}

static const char kernel_function[] = R"(
__device__ float real_exp(float x) { return ::expf(x); }

__device__ double real_exp(double x) { return ::exp(x); }

__device__ float real_log(float x) { return ::logf(x); }

__device__ double real_log(double x) { return ::log(x); }

__device__ float real_min(float x, float y) { return ::fminf(x, y); }

__device__ double real_min(double x, double y) { return ::fmin(x, y); }

__device__ float real_max(float x, float y) { return ::fmaxf(x, y); }

__device__ double real_max(double x, double y) { return ::fmax(x, y); }

)";

static const char kernel_elementwise_template[] = R"(

extern "C" __global__ void $name($parameter){
  for(int idx = blockIdx.x * blockDim.x + threadIdx.x;
      idx < N;
      idx += gridDim.x * blockDim.x) {
      $compute
}
}
)";

}  // namespace ir
}  // namespace framework
}  // namespace paddle
