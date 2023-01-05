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

#include "paddle/fluid/framework/new_executor/standalone_executor.h"

namespace paddle {
namespace operators {

static void BuildScopeForControlFlowOp(
    const framework::InterpreterCore &interpreter_core,
    const framework::BlockDesc &block,
    framework::Scope *scope) {
  for (auto &var_desc : block.AllVars()) {
    auto var_name = var_desc->Name();
    if (var_name == framework::kEmptyVarName) {
      continue;
    }
    VLOG(5) << "[BuildScopeForControlFlowOp]"
            << "start:" << var_name;
    if (var_desc->Persistable()) {
      VLOG(5) << "[BuildScopeForControlFlowOp]"
              << "Don't process persistent: " << var_name;
    } else {
      auto *ptr = scope->Var(var_name);
      InitializeVariable(ptr, var_desc->GetType());
      VLOG(5) << "[BuildScopeForControlFlowOp]"
              << "Not Found locally and created: " << var_name;
    }
  }

  auto &data_transfer_added_vars =
      interpreter_core.GetVariableScope()->DataTransferAddedVars();
  for (size_t i = 0; i < data_transfer_added_vars.size(); i++) {
    auto *ptr = scope->Var(data_transfer_added_vars[i].first);
    InitializeVariable(ptr,
                       static_cast<paddle::framework::proto::VarType::Type>(
                           data_transfer_added_vars[i].second));
    VLOG(5) << "[BuildScopeForControlFlowOp]"
            << "Initialize Transfer Added Variable "
            << data_transfer_added_vars[i].first;
  }
}

}  // namespace operators
}  // namespace paddle
