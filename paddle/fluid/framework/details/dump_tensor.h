// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <map>
#include <string>
#include <vector>

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/imperative/type_defs.h"
#include "paddle/fluid/imperative/var_helper.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace framework {
namespace details {

void DumpTensor2File(const framework::LoDTensor& tensor,
                     const std::string& folder_path,
                     const std::string& var_name);

void DumpTensor(const std::string& op_type, const std::string& var_name,
                const std::string& value_type, /* inputs or outputs*/
                const framework::Variable* var, const platform::Place& place);

void DumpTensor(const framework::OperatorBase& op,
                const framework::Scope& exec_scope,
                const platform::Place& place);

template <typename VarType>
void DumpTensorInDygraph(const std::string& op_type,
                         const imperative::NameVarMap<VarType>& op_ins,
                         const imperative::NameVarMap<VarType>& op_outs,
                         platform::Place place) {
  for (const auto& pair : op_ins) {
    for (const auto& ivar : pair.second) {
      auto* var = ivar->MutableVar();
      if (var == nullptr) continue;
      DumpTensor(op_type, paddle::imperative::GetNameFromVar(ivar), "InputVars",
                 var, place);
    }
  }

  for (const auto& pair : op_outs) {
    for (const auto& ivar : pair.second) {
      auto* var = ivar->MutableVar();
      if (var == nullptr) continue;
      DumpTensor(op_type, paddle::imperative::GetNameFromVar(ivar),
                 "OutputVars", var, place);
    }
  }
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
