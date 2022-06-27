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

#include "paddle/fluid/jit/function_utils.h"

#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace jit {
namespace utils {
void FetchVarsByNames(const std::vector<std::string> &names,
                      const framework::Scope &scope,
                      std::vector<Variable> *outs) {
  for (auto &out_name : names) {
    VLOG(3) << "fetch out: " << out_name;
    auto *var = scope.FindVar(out_name);
    auto &src_tensor = var->Get<DenseTensor>();
    Variable v;
    auto *p = v.GetMutable<DenseTensor>();
    *p = src_tensor;
    outs->emplace_back(v);
  }
}

void ShareInputsIntoScope(const std::vector<std::string> &ordered_input_names,
                          const std::vector<Variable> &vars,
                          framework::Scope *scope) {
  VLOG(3) << "vars size: " << vars.size();
  PADDLE_ENFORCE_EQ(
      vars.size(),
      ordered_input_names.size(),
      platform::errors::InvalidArgument(
          "vars.size() should be equal to ordered_input_names.size()."));

  for (size_t i = 0; i < vars.size(); i++) {
    VLOG(3) << "share into scope: " << ordered_input_names[i];
    auto &dense_tensor = vars[i].Get<DenseTensor>();
    auto *var = scope->Var(ordered_input_names[i]);
    auto *dst_tensor = var->GetMutable<DenseTensor>();
    *dst_tensor = dense_tensor;
  }
}

void ShareParamsIntoScope(const std::vector<std::string> &param_names,
                          const Name2VariableMap &params_dict,
                          framework::Scope *scope) {
  VLOG(3) << "param_names size: " << param_names.size();
  for (size_t i = 0; i < param_names.size(); ++i) {
    std::string name = param_names[i];
    auto &param = params_dict.find(name)->second;
    auto &dense_tensor = param.Get<DenseTensor>();
    VLOG(3) << "share into scope: " << name;
    auto *var = scope->Var(name);
    auto *dst_tensor = var->GetMutable<DenseTensor>();
    *dst_tensor = dense_tensor;
  }
}

void RemoveFeedFetch(framework::ProgramDesc *program_desc) {
  for (size_t i = 0; i < program_desc->Size(); ++i) {
    auto *block = program_desc->MutableBlock(i);
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

}  // namespace utils
}  // namespace jit
}  // namespace paddle
