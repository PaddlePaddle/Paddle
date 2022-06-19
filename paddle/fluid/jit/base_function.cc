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

void FunctionSchema::AddInputArg(std::string name, bool is_output) {
  input_args.emplace_back(name, is_output);
}

void FunctionSchema::AddOutputArg(std::string name, bool is_output) {
  output_args.emplace_back(name, is_output);
}

BaseFunction::BaseFunction(
    const framework::ProgramDesc &program_desc,
    const std::vector<std::string> param_names_for_program,
    const VariableNameMap &params_dict)
    : program_desc_(program_desc) {
  // Parse FunctionSchema
  // skip_var_name_ = program_desc_.GetFetchTargetNames();
  for (auto &in_name : program_desc_.GetFeedTargetNames()) {
    schema_.AddInputArg(in_name, false);
  }
  for (auto &out_name : program_desc_.GetFetchTargetNames()) {
    schema_.AddOutputArg(out_name, true);
  }
  // share params into scope
  SharePartialIntoScope(param_names_for_program, params_dict);
  VLOG(6) << framework::GenScopeTreeDebugInfo(&scope_);
  // remove feed fetch op
  RemoveFeedFetch();
}

void BaseFunction::FetchOutput(std::vector<Variable> *outs) {
  for (auto &out_name : schema_.GetOutputArgNames()) {
    VLOG(3) << "fetch out: " << out_name;
    auto *var = scope_.FindVar(out_name);
    auto &src_tensor = var->Get<phi::DenseTensor>();
    Variable v;
    auto *p = v.GetMutable<DenseTensor>();
    *p = src_tensor;
    outs->emplace_back(v);
  }
}

void BaseFunction::ShareIntoScope(const VariableNameMap &ivals) {
  VLOG(3) << "ivals size: " << ivals.size();
  for (auto it = ivals.begin(); it != ivals.end(); ++it) {
    VLOG(3) << "share into scope: " << it->first;
    DenseTensor dense_tensor = it->second.Get<DenseTensor>();
    auto *var = scope_.Var(it->first);
    auto *dst_tensor = var->GetMutable<DenseTensor>();
    *dst_tensor = dense_tensor;
  }
}

void BaseFunction::SharePartialIntoScope(
    const std::vector<std::string> param_names_for_program,
    const VariableNameMap &params_dict) {
  VLOG(3) << "ivals size: " << param_names_for_program.size();
  for (size_t i = 0; i < param_names_for_program.size(); ++i) {
    std::string name = param_names_for_program[i];
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
      if (op->Type() == "feed" || op->Type() == "fetch") {
        VLOG(3) << "remove op type: " << op->Type() << ", index: " << i;
        block->RemoveOp(i, i + 1);
      }
    }
  }
}

}  // namespace jit
}  // namespace paddle
