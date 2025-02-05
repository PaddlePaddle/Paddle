// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/fuse_pass_base.h"

namespace paddle {
namespace framework {
namespace ir {

class Graph;

class InplaceOpVarPass : public FusePassBase {
 protected:
  using FusePassBase::ApplyImpl;
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  virtual ~InplaceOpVarPass() = default;

  int ApplyImpl(ir::Graph* graph,
                const std::unordered_set<std::string>& deny_var_names) const;

  bool IsValidInplaceOp(
      Node* node, const std::unordered_set<std::string>& deny_var_names) const;

  std::vector<std::string> GetControlFlowVarNames(ir::Graph* graph) const;

  std::set<std::string> inplace_ops_{"reshape",
                                     "reshape2",
                                     "unsqueeze",
                                     "unsqueeze2",
                                     "squeeze",
                                     "squeeze2",
                                     "flatten_contiguous_range"};
  std::set<std::string> control_flow_ops_{"while", "conditional_block"};
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
