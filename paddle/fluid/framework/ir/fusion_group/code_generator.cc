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

#include "paddle/fluid/framework/ir/fusion_group/code_generator.h"
#include <sstream>
#include <unordered_set>
#include "paddle/fluid/framework/ir/fusion_group/code_generator_helper.h"
#include "paddle/fluid/framework/ir/fusion_group/cuda_resources.h"
#include "paddle/fluid/framework/ir/fusion_group/operation.h"

namespace paddle {
namespace framework {
namespace ir {
namespace fusion_group {

CodeGenerator::CodeGenerator() {
  // Only support elementwise operations now.
  code_templates_.resize(1);

  CodeTemplate elementwise_t(cuda_kernel_template_1d);
  code_templates_[0] = elementwise_t;
}

std::string CodeGenerator::Generate(SubGraph* subgraph) {
  std::vector<OperationExpression> expressions = ConvertToExpressions(subgraph);
  return Generate(subgraph->GetFuncName(), subgraph->GetDataType(),
                  expressions);
}

static bool HasInput(Node* n, std::string name) {
  PADDLE_ENFORCE_EQ(n && n->IsOp() && n->Op(), true,
                    platform::errors::InvalidArgument(
                        "Expected node %p to be an operator node.", n));
  std::vector<std::string> input_names = n->Op()->InputNames();
  std::unordered_set<std::string> input_names_set(input_names.begin(),
                                                  input_names.end());
  return input_names_set.find(name) != input_names_set.end();
}

std::vector<OperationExpression> CodeGenerator::ConvertToExpressions(
    SubGraph* subgraph) {
  std::unordered_map<std::string, int> var_ids = EncodeVarNodes(subgraph);
  std::vector<OperationExpression> expressions;
  for (auto* node : subgraph->SortedNodes()) {
    if (node && node->IsOp() && node->Op()) {
      auto* op = node->Op();

      // Input ids should be set in fixed order, like:
      //  - X, Y in forward operations
      //  - X, Y, Out, out@GRAD in backward operations
      std::vector<int> input_ids;
      std::vector<std::string> input_names =
          OperationMap::Instance().Get(op->Type()).input_names;
      for (auto& name : input_names) {
        // Some input vars are not used in grad ops, such as
        // "elementwise_add_grad", where "X", "Y" and "Out" are not used.
        if (HasInput(node, name) && op->Input(name).size() >= 1U) {
          // TODO(liuyiqun): support duplicated input.
          PADDLE_ENFORCE_NE(
              var_ids.find(op->Input(name)[0]), var_ids.end(),
              platform::errors::InvalidArgument(
                  "Input(%s) of operation %s is not set.", name, op->Type()));
          input_ids.push_back(var_ids[op->Input(name)[0]]);
        } else {
          input_ids.push_back(-1);
        }
      }
      // Output ids should be set in fixed order, like:
      //  - dx, dy in backward operations
      std::vector<int> output_ids;
      std::vector<std::string> output_names =
          OperationMap::Instance().Get(op->Type()).output_names;
      for (auto& name : output_names) {
        PADDLE_ENFORCE_EQ(
            op->Output(name).size(), 1U,
            platform::errors::InvalidArgument(
                "Output(%s) of operation %s is not set.", name, op->Type()));
        PADDLE_ENFORCE_NE(
            var_ids.find(op->Output(name)[0]), var_ids.end(),
            platform::errors::InvalidArgument(
                "Output(%s) of operation %s is not set.", name, op->Type()));
        output_ids.push_back(var_ids[op->Output(name)[0]]);
      }
      expressions.push_back(
          OperationExpression(node->Name(), input_ids, output_ids));
    }
  }
  return expressions;
}

// In order to get the right result of expression, we need to calculate and
// store the expression as suffix Expressions using vector.
std::string CodeGenerator::Generate(
    std::string func_name, std::string dtype,
    const std::vector<OperationExpression>& expressions) {
  // TODO(liuyiqun): Check whether all expressions are elementwise operations.
  std::set<int> input_ids = DistilInputIds(expressions);
  std::set<int> output_ids = DistilOutputIds(expressions);

  TemplateVariable template_var;
  template_var.Add("func_name", func_name);
  template_var.Add("parameters", EmitParameters(input_ids, output_ids, dtype));
  template_var.Add("compute_body",
                   EmitComputeBody(expressions, input_ids, output_ids, dtype));

  std::string predefined_cuda_functions;
  if (dtype == "float") {
    predefined_cuda_functions = predefined_cuda_functions_fp32;
  } else if (dtype == "double") {
    predefined_cuda_functions = predefined_cuda_functions_fp64;
  } else if (dtype == "float16") {
    predefined_cuda_functions = predefined_cuda_functions_fp16;
  }
  return predefined_cuda_functions + code_templates_[0].Format(template_var);
}

std::set<int> CodeGenerator::DistilInputIds(
    const std::vector<OperationExpression>& expressions) {
  std::set<int> input_ids;
  // Use std::set to remove the reptead id and get a ordered list.
  for (size_t i = 0; i < expressions.size(); i++) {
    for (auto id : expressions[i].GetInputIds()) {
      if (id >= 0) {
        input_ids.insert(id);
      }
    }
  }
  return input_ids;
}

std::set<int> CodeGenerator::DistilOutputIds(
    const std::vector<OperationExpression>& expressions) {
  std::set<int> output_ids;
  // Use std::set to remove the reptead id and get a ordered list.
  for (size_t i = 0; i < expressions.size(); i++) {
    for (auto id : expressions[i].GetOutputIds()) {
      output_ids.insert(id);
    }
  }
  return output_ids;
}

// we get the parameter list code for the expression information
std::string CodeGenerator::EmitParameters(const std::set<int>& input_ids,
                                          const std::set<int>& output_ids,
                                          std::string dtype) {
  std::stringstream ret;
  ret << "int N, ";

  // If a id is in the input and output list at the same time, then remove it
  // from the input list.
  for (auto id : input_ids) {
    if (output_ids.find(id) == output_ids.end()) {
      ret << dtype << "* " << ArgName(id) << ", ";
    }
  }

  size_t index = 0;
  for (auto id : output_ids) {
    ret << dtype << "* " << ArgName(id);
    if (index != output_ids.size() - 1) {
      ret << ", ";
    }
    index++;
  }

  return ret.str();
}

std::string CodeGenerator::EmitComputeBody(
    const std::vector<OperationExpression>& expressions,
    const std::set<int>& input_ids, const std::set<int>& output_ids,
    std::string dtype) {
  std::ostringstream compute;
  std::unordered_set<int> used;
  std::string compute_dtype = (dtype == "float16") ? "float" : dtype;
  for (size_t i = 0; i < expressions.size(); i++) {
    VLOG(3) << DebugString(expressions[i]);
    compute << expressions[i].GetExpression(compute_dtype, &used);
  }

  // Load input to temporal variables.
  std::ostringstream load;
  for (auto id : input_ids) {
    if (output_ids.find(id) == output_ids.end() &&
        used.find(id) != used.end()) {
      if (dtype == "float16") {
        load << "float " << TmpName(id) << " = __half2float(" << ArgName(id)
             << "[idx]);";
      } else {
        load << dtype << " " << TmpName(id) << " = " << ArgName(id) << "[idx];";
      }
    }
  }

  // Store temporal variables to memory.
  std::ostringstream store;
  for (auto id : output_ids) {
    if (dtype == "float16") {
      store << ArgName(id) << "[idx] = __float2half(" << TmpName(id) << ");";
    } else {
      store << ArgName(id) << "[idx] = " << TmpName(id) << ";";
    }
  }

  return load.str() + compute.str() + store.str();
}

std::unordered_map<std::string, int> CodeGenerator::EncodeVarNodes(
    SubGraph* subgraph) {
  const auto& input_var_nodes = subgraph->GetInputVarNodes();
  const auto& output_var_nodes = subgraph->GetOutputVarNodes();

  int id = 0;
  std::unordered_map<std::string, int> var_ids;
  // Numbering input vars.
  for (auto* in : input_var_nodes) {
    VLOG(3) << "Encoding input names:" << in->Name() << ", id:" << id;
    if (var_ids.find(in->Name()) == var_ids.end()) {
      var_ids[in->Name()] = id++;
    }
  }
  // Numbering internal vars.
  for (auto* node : subgraph->SortedNodes()) {
    if (node && node->IsVar() && node->Var()) {
      bool is_found = false;
      for (auto* in : input_var_nodes) {
        if (node == in) {
          is_found = true;
          break;
        }
      }
      if (is_found) {
        continue;
      }
      for (auto* out : output_var_nodes) {
        if (node == out) {
          is_found = true;
          break;
        }
      }
      PADDLE_ENFORCE_EQ(
          is_found, true,
          platform::errors::Unimplemented(
              "Subgraph with internal var nodes (%s) is not supported yet.",
              node->Name()));
    }
  }
  // Encoding output vars.
  for (auto* out : output_var_nodes) {
    VLOG(3) << "Ecoding output names:" << out->Name() << ", id:" << id;
    if (var_ids.find(out->Name()) == var_ids.end()) {
      var_ids[out->Name()] = id++;
    }
  }
  return var_ids;
}

}  // namespace fusion_group
}  // namespace ir
}  // namespace framework
}  // namespace paddle
