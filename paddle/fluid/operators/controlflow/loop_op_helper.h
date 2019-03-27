// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/variant.h"

namespace paddle {
namespace operators {

static constexpr char kStepBlock[] = "sub_block";
static constexpr char kSkipEagerDeletionVars[] = "skip_eager_deletion_vars";

namespace recurrent {
constexpr char kInputs[] = "inputs";
constexpr char kInitialStates[] = "initial_states";
constexpr char kParameters[] = "parameters";
constexpr char kOutputs[] = "outputs";
constexpr char kStepScopes[] = "step_scopes";
constexpr char kExStates[] = "ex_states";
constexpr char kStates[] = "states";
constexpr char kStepBlock[] = "sub_block";
constexpr char kReverse[] = "reverse";
constexpr char kIsTrain[] = "is_train";
constexpr char kInputGrads[] = "inputs@GRAD";
constexpr char kOutputGrads[] = "outputs@GRAD";
constexpr char kParamGrads[] = "parameters@GRAD";
constexpr char kInitStateGrads[] = "initial_states@GRAD";
}  // namespace recurrent

void PrepareSafeEagerDeletionOnLoopOps(
    int block_id,
    const std::vector<std::unique_ptr<framework::OperatorBase>> &all_ops);

void PrepareSafeEagerDeletionOnLoopOps(
    const std::vector<framework::OperatorBase *> &while_ops,
    const std::vector<framework::OperatorBase *> &while_grad_ops,
    const std::vector<framework::OperatorBase *> &recurrent_ops,
    const std::vector<framework::OperatorBase *> &recurrent_grad_ops);

inline std::string GetSkipEagerDeletionVarsDebugString(
    const std::vector<std::string> &vars) {
  std::string str = "Skip " + std::to_string(vars.size()) +
                    " var(s) in eager deletion mode: ";
  for (auto &var : vars) {
    str.append(var);
    str.push_back(' ');
  }
  return str;
}

}  // namespace operators
}  // namespace paddle
