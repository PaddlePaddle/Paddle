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

std::string ExtractDataType(const std::vector<Node*>& nodes) {
  std::string dtype_str = "";
  for (const auto* n : nodes) {
    if (n && n->IsVar() && n->Var()) {
      // The data type of all inputs/outputs must be the same, which are
      //  checked when detecting the subgraph.
      auto dtype = n->Var()->GetDataType();
      if (dtype == proto::VarType::FP32) {
        dtype_str = "float";
      } else if (dtype == proto::VarType::FP64) {
        dtype_str = "double";
      } else if (dtype == proto::VarType::FP16) {
        dtype_str = "__half";
      }
      break;
    }
  }

  return dtype_str;
}

CodeGenerator::CodeGenerator() {
  // Only support elementwise operations now.
  code_templates_.resize(1);

  CodeTemplate elementwise_t(cuda_kernel_template_1d);
  code_templates_[0] = elementwise_t;
}

std::string CodeGenerator::Generate(SubGraph* subgraph) {
  std::vector<OperationExpression> expressions = ConvertToExpressions(subgraph);
  return Generate(subgraph->GetFuncName(), expressions);
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
  std::vector<Node*> intermediate_out_nodes =
      subgraph->GetIntermediateOutVarNodes();
  std::vector<OperationExpression> expressions;
  for (auto* node : subgraph->SortedNodes()) {
    if (node && node->IsOp() && node->Op()) {
      auto* op = node->Op();
      AttributeMap attr = *(op->MutableAttrMap());

      // Input ids should be set in fixed order, like:
      //  - X, Y in forward operations
      //  - X, Y, Out, out@GRAD in backward operations
      std::vector<int> input_ids;
      std::string op_name = op->Type();
      auto operation = OperationMap::Instance().Get(op_name);
      std::vector<std::string> input_names = operation.input_names;

      for (auto& name : input_names) {
        // Some input vars are not used in grad ops, such as
        // "elementwise_add_grad", where "X", "Y" and "Out" are not used.
        if ((HasInput(node, name) && op->Input(name).size() >= 1U)) {
          for (size_t i = 0; i < op->Input(name).size(); i++) {
            PADDLE_ENFORCE_NE(
                var_ids.find(op->Input(name)[i]), var_ids.end(),
                platform::errors::InvalidArgument(
                    "Input(%s) of operation %s is not set.", name, op->Type()));
            input_ids.push_back(var_ids[op->Input(name)[i]]);
          }
        } else {
          input_ids.push_back(-1);
        }
      }

      // Output ids should be set in fixed order, like:
      //  - dx, dy in backward operations
      std::vector<int> output_ids;
      std::vector<std::string> output_names =
          OperationMap::Instance().Get(op->Type()).output_names;
      std::unordered_map<int, bool> intermediate_state;

      for (auto& name : output_names) {
        PADDLE_ENFORCE_NE(
            var_ids.find(op->Output(name)[0]), var_ids.end(),
            platform::errors::InvalidArgument(
                "Output(%s) of operation %s is not set.", name, op->Type()));
        output_ids.push_back(var_ids[op->Output(name)[0]]);
        bool enable_intermediate = false;
        for (auto* n : intermediate_out_nodes) {
          if (n->Name() == op->Output(name)[0]) {
            enable_intermediate = true;
            break;
          }
        }
        intermediate_state[var_ids[op->Output(name)[0]]] = enable_intermediate;
      }

      std::string lhs_type = ExtractDataType(node->outputs);
      std::string rhs_type = ExtractDataType(node->inputs);
      auto expression =
          OperationExpression(node->Name(), input_ids, output_ids, rhs_type,
                              lhs_type, intermediate_state);
      expression.SetAttr(attr);
      expressions.push_back(expression);
    }
  }
  return expressions;
}

// In order to get the right result of expression, we need to calculate and
// store the expression as suffix Expressions using vector.
std::string CodeGenerator::Generate(
    std::string func_name,
    const std::vector<OperationExpression>& expressions) {
  // TODO(liuyiqun): Check whether all expressions are elementwise operations.
  std::set<int> input_ids = std::move(DistilInputIds(expressions));
  std::set<int> output_ids = std::move(DistilOutputIds(expressions));
  std::set<int> intermediate_ids =
      std::move(DistilIntermediateIds(expressions));
  std::unordered_map<int, std::string> dtypes =
      std::move(DistilDtypes(expressions));
  TemplateVariable template_var;
  template_var.Add("func_name", func_name);
  template_var.Add("parameters", EmitParameters(input_ids, output_ids,
                                                intermediate_ids, dtypes));
  template_var.Add("compute_body",
                   EmitComputeBody(expressions, input_ids, output_ids,
                                   intermediate_ids, dtypes));

  std::set<std::string> all_dtype;
  for (const auto& type : dtypes) {
    all_dtype.insert(type.second);
  }
  std::string predefined_cuda_functions = "";
  if (all_dtype.find("float") != all_dtype.end() &&
      all_dtype.find("__half") == all_dtype.end()) {
    predefined_cuda_functions += predefined_cuda_functions_fp32;
  }
  if (all_dtype.find("double") != all_dtype.end()) {
    predefined_cuda_functions += predefined_cuda_functions_fp64;
  }
  if (all_dtype.find("__half") != all_dtype.end()) {
    predefined_cuda_functions += predefined_cuda_functions_fp16;
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

std::set<int> CodeGenerator::DistilIntermediateIds(
    const std::vector<OperationExpression>& expressions) {
  std::set<int> intermediate_ids;
  // Use std::set to remove the reptead id and get a ordered list.
  for (size_t i = 0; i < expressions.size(); i++) {
    for (auto id : expressions[i].GetOutputIds()) {
      auto intermediate_state = expressions[i].GetIntermediateState();
      if (intermediate_state.find(id) != intermediate_state.end() &&
          intermediate_state[id]) {
        intermediate_ids.insert(id);
      }
    }
  }
  return intermediate_ids;
}

std::unordered_map<int, std::string> CodeGenerator::DistilDtypes(
    const std::vector<OperationExpression>& expressions) {
  std::unordered_map<int, std::string> dtypes;
  for (const auto& expression : expressions) {
    for (auto id : expression.GetInputIds()) {
      auto dtype = expression.GetRHSType();
      if (dtypes.find(id) == dtypes.end()) {
        dtypes[id] = dtype;
      } else {
        PADDLE_ENFORCE_EQ(
            dtypes[id], dtype,
            platform::errors::PreconditionNotMet(
                "In fusion group, Same Node id must have same date type"));
      }
    }
    for (auto id : expression.GetOutputIds()) {
      auto dtype = expression.GetLHSType();
      if (dtypes.find(id) == dtypes.end()) {
        dtypes[id] = dtype;
      } else {
        PADDLE_ENFORCE_EQ(
            dtypes[id], dtype,
            platform::errors::PreconditionNotMet(
                "In fusion group, Same Node id must have same date type"));
      }
    }
  }
  return dtypes;
}

// we get the parameter list code for the expression information
std::string CodeGenerator::EmitParameters(
    const std::set<int>& input_ids, const std::set<int>& output_ids,
    const std::set<int>& intermediate_ids,
    const std::unordered_map<int, std::string>& dtypes) const {
  std::stringstream ret;
  ret << "int N, ";

  // If a id is in the input and output list at the same time, then remove it
  // from the input list.
  for (auto id : input_ids) {
    if (output_ids.find(id) == output_ids.end()) {
      ret << "const " << dtypes.at(id) << "* __restrict__ " << ArgName(id)
          << ", ";
    }
  }

  size_t index = 0;
  std::vector<std::string> output_args;
  for (auto id : output_ids) {
    if (intermediate_ids.find(id) == intermediate_ids.end()) {
      std::string args_str = dtypes.at(id) + "* " + ArgName(id);
      output_args.push_back(args_str);
    }
  }
  for (auto args : output_args) {
    ret << args;
    if (index != output_args.size() - 1) {
      ret << ", ";
    }
    index++;
  }
  return ret.str();
}

std::string CodeGenerator::EmitComputeBody(
    const std::vector<OperationExpression>& expressions,
    const std::set<int>& input_ids, const std::set<int>& output_ids,
    const std::set<int>& intermediate_ids,
    const std::unordered_map<int, std::string>& dtypes) const {
  std::ostringstream compute;
  std::unordered_set<int> used;
  for (size_t i = 0; i < expressions.size(); i++) {
    VLOG(3) << DebugString(expressions[i]);
    compute << expressions[i].GetExpression(&used);
  }

  // Load input to temporal variables.
  std::ostringstream load;
  for (auto id : input_ids) {
    if (output_ids.find(id) == output_ids.end() &&
        used.find(id) != used.end()) {
      load << dtypes.at(id) << " " << TmpName(id) << " = "
           << "__ldg(&" << VarName(id) << ")"
           << ";";
    }
  }
  // Store temporal variables to memory.
  std::ostringstream store;
  for (auto id : output_ids) {
    if (intermediate_ids.find(id) == intermediate_ids.end()) {
      store << VarName(id) << " = " << TmpName(id) << ";";
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
