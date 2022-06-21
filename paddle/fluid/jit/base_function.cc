// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/jit/base_function.h"

namespace paddle {
namespace jit {

Argument::Argument(const std::string &name, bool is_out)
    : name_(name), is_output_(is_out) {}

const std::string &Argument::Name() const { return name_; }

std::vector<std::string> FunctionSchema::GetInputArgNames() {
  std::vector<std::string> input_arg_names;
  for (auto &arg : input_args) {
    input_arg_names.emplace_back(arg.Name());
  }
  return input_arg_names;
}

std::vector<std::string> FunctionSchema::GetOutputArgNames() {
  std::vector<std::string> output_arg_names;
  for (auto &arg : output_args) {
    output_arg_names.emplace_back(arg.Name());
  }
  return output_arg_names;
}

void FunctionSchema::AddInputArg(std::string name) {
  input_args.emplace_back(name, false);
}

void FunctionSchema::AddOutputArg(std::string name) {
  output_args.emplace_back(name, true);
}

BaseFunction::BaseFunction(const framework::ProgramDesc &program_desc,
                           const std::vector<std::string> &param_names,
                           const VariableNameMap &params_dict,
                           const phi::Place &place)
    : program_desc_(program_desc), place_(place) {
  // Parse FunctionSchema
  for (auto &in_name : program_desc_.GetFeedTargetNames()) {
    schema_.AddInputArg(in_name);
  }
  for (auto &out_name : program_desc_.GetFetchTargetNames()) {
    schema_.AddOutputArg(out_name);
  }
  // share params into scope
  ShareParamsIntoScope(param_names, params_dict);
  VLOG(6) << framework::GenScopeTreeDebugInfo(&scope_);
  // remove feed fetch op
  RemoveFeedFetch();
}

void BaseFunction::FetchOutput(std::vector<Variable> *outs) {
  for (auto &out_name : schema_.GetOutputArgNames()) {
    VLOG(3) << "fetch out: " << out_name;
    auto *var = scope_.FindVar(out_name);
    VLOG(3) << "after scope_.FindVar(out_name);";
    auto &src_tensor = var->Get<phi::DenseTensor>();
    VLOG(3) << "var->Get<phi::DenseTensor>();";
    Variable v;
    auto *p = v.GetMutable<DenseTensor>();
    *p = src_tensor;
    outs->emplace_back(v);
  }
}

void BaseFunction::ShareInputsIntoScope(const std::vector<Variable> &vars) {
  VLOG(3) << "vars size: " << vars.size();
  std::vector<std::string> ordered_input_names = schema_.GetInputArgNames();
  PADDLE_ENFORCE_EQ(
      vars.size(), ordered_input_names.size(),
      platform::errors::InvalidArgument(
          "vars.size() should be equal to ordered_input_names.size()."));

  for (size_t i = 0; i < vars.size(); i++) {
    VLOG(3) << "share into scope: " << ordered_input_names[i];
    auto &dense_tensor = vars[i].Get<DenseTensor>();
    auto *var = scope_.Var(ordered_input_names[i]);
    auto *dst_tensor = var->GetMutable<DenseTensor>();
    *dst_tensor = dense_tensor;
  }
}

void BaseFunction::ShareParamsIntoScope(
    const std::vector<std::string> &param_names,
    const VariableNameMap &params_dict) {
  VLOG(3) << "param_names size: " << param_names.size();
  for (size_t i = 0; i < param_names.size(); ++i) {
    std::string name = param_names[i];
    Variable val = params_dict.find(name)->second;
    auto &dense_tensor = val.Get<DenseTensor>();
    VLOG(3) << "share into scope: " << name;
    auto *var = scope_.Var(name);
    auto *dst_tensor = var->GetMutable<DenseTensor>();
    *dst_tensor = dense_tensor;
  }
}

void BaseFunction::RemoveFeedFetch() {
  for (size_t i = 0; i < program_desc_.Size(); ++i) {
    auto *block = program_desc_.MutableBlock(i);
    const auto &all_ops = block->AllOps();
    size_t op_size = all_ops.size();
    VLOG(3) << "op_size: " << op_size;
    for (int i = op_size - 1; i >= 0; i--) {
      auto op = all_ops[i];
      if (op->Type() == "feed") {
        VLOG(3) << "remove op type: " << op->Type() << ", index: " << i
                << ", var name: " << op->Input("X")[0];
        block->RemoveVar(op->Input("X")[0]);
        block->RemoveOp(i, i + 1);
      } else if (op->Type() == "fetch") {
        VLOG(3) << "remove op type: " << op->Type() << ", index: " << i
                << ", var name: " << op->Output("Out")[0];
        block->RemoveVar(op->Output("Out")[0]);
        block->RemoveOp(i, i + 1);
      }
    }
  }
}

}  // namespace jit
}  // namespace paddle
