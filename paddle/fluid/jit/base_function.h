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

#pragma once

#include <ostream>
#include <string>

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/jit/ivalue.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/utils/none.h"
#include "paddle/utils/optional.h"

namespace paddle {
namespace jit {

class Argument {
 public:
  explicit Argument(const std::string &name, bool is_out = false)
      : name_(name), is_output_(is_out) {}

  const std::string &Name() const { return name_; }

 private:
  std::string name_;
  // paddle::optional<IValue> default_val_;
  bool is_output_;
};

class FunctionSchema {
 public:
  FunctionSchema() = default;

  std::vector<std::string> GetInputArgNames() {
    std::vector<std::string> input_arg_names;
    for (auto &arg : input_args) {
      input_arg_names.emplace_back(arg.Name());
    }
    return input_arg_names;
  }

  std::vector<std::string> GetOutputArgNames() {
    std::vector<std::string> output_arg_names;
    for (auto &arg : output_args) {
      output_arg_names.emplace_back(arg.Name());
    }
    return output_arg_names;
  }

  void AddInputArg(std::string name, bool is_output) {
    input_args.emplace_back(name, is_output);
  }

  void AddOutputArg(std::string name, bool is_output) {
    output_args.emplace_back(name, is_output);
  }

 private:
  std::vector<Argument> input_args;
  std::vector<Argument> output_args;
};

// TODO(dev): make it as abstract class
class BaseFunction {
 public:
  BaseFunction(const framework::ProgramDesc &program_desc,
               const std::vector<std::string> param_names_for_program,
               const std::map<std::string, IValue> &params_dict)
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
  virtual ~BaseFunction() {}

  virtual std::vector<IValue> operator()(const std::vector<IValue> &inputs) = 0;

 protected:
  void FetchOutput(std::vector<IValue> *outs) {
    for (auto &out_name : schema_.GetOutputArgNames()) {
      VLOG(3) << "fetch out: " << out_name;
      auto *var = scope_.FindVar(out_name);
      auto &src_tensor = var->Get<phi::DenseTensor>();
      Tensor t(std::make_shared<phi::DenseTensor>());
      auto *dst_tensor = const_cast<phi::DenseTensor *>(
          dynamic_cast<const phi::DenseTensor *>(t.impl().get()));
      *dst_tensor = src_tensor;
      outs->emplace_back(t);
    }
  }

  void ShareIntoScope(const std::vector<IValue> &ivals) {
    VLOG(3) << "ivals size: " << ivals.size();
    for (size_t i = 0; i < ivals.size(); ++i) {
      auto &tensor = ivals[i].AsTensor();
      VLOG(3) << "share into scope: " << tensor.name();
      auto *var = scope_.Var(tensor.name());
      auto *dst_tensor = var->GetMutable<phi::DenseTensor>();
      auto t = std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
      *dst_tensor = *t;
    }
  }

  void SharePartialIntoScope(
      const std::vector<std::string> param_names_for_program,
      const std::map<std::string, IValue> &params_dict) {
    VLOG(3) << "ivals size: " << param_names_for_program.size();
    for (size_t i = 0; i < param_names_for_program.size(); ++i) {
      std::string name = param_names_for_program[i];
      IValue val = params_dict.find(name)->second;
      auto &tensor = val.AsTensor();
      VLOG(3) << "share into scope: " << tensor.name();
      auto *var = scope_.Var(tensor.name());
      auto *dst_tensor = var->GetMutable<phi::DenseTensor>();
      auto t = std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
      *dst_tensor = *t;
    }
  }

  void RemoveFeedFetch() {
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

 protected:
  framework::ProgramDesc program_desc_;
  // TODO(dev): need a better way to share params
  // std::vector<IValue> &param_for_program_;
  // std::vector<std::string> skip_var_name_;
  FunctionSchema schema_;
  // global_scope place params
  framework::Scope scope_;
  //   framework::Executor inner_exe_;
};

}  // namespace jit
}  // namespace paddle
