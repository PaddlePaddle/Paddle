/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#include <string>
#include <unordered_map>
#include <unordered_set>

#include "paddle/framework/operator.h"
#include "paddle/framework/program_desc.h"

namespace paddle {
namespace framework {

// Create the backward operator from a forward operator.
// TODO(yuyang18): Add more API reference comment.
extern std::unique_ptr<OperatorBase> Backward(
    const OperatorBase& forwardOp,
    const std::unordered_set<std::string>& no_grad_vars);

struct GradVarInfo {
  GradVarInfo() {}
  GradVarInfo(const std::string& name, int block_idx, int op_idx)
      : name_(name), block_idx_(block_idx), op_idx_(op_idx) {}

  bool operator==(const GradVarInfo& b) const {
    return name_ == b.name_ && block_idx_ == b.block_idx_ &&
           op_idx_ == b.op_idx_;
  }

  std::string name_;
  int block_idx_;
  int op_idx_;
};

using ParamGradInfoMap = std::unordered_map<std::string /*fwd_var_name*/,
                                            GradVarInfo /*grad_var_info*/>;

ParamGradInfoMap AppendBackward(
    ProgramDescBind& program_desc, const VarDescBind& target,
    const std::unordered_set<std::string>& no_grad_vars);

}  // namespace framework
}  // namespace paddle
