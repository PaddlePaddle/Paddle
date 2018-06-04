// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/details/ssa_graph_builder.h"
namespace paddle {
namespace framework {
namespace details {

class FuseAllReduceGraphBuilder : public SSAGraphBuilder {
 public:
  explicit FuseAllReduceGraphBuilder(std::unique_ptr<SSAGraphBuilder>&& builder)
      : builder_(std::move(builder)) {}
  std::unique_ptr<SSAGraph> Build(const ProgramDesc& program) const override;

 private:
  /**
   * Get All-Reduce operator into multiple sets.
   * The order of set is the order of execution.
   */
  std::vector<std::unordered_set<std::unique_ptr<OpHandleBase>>>
  GetNotDependedAllReduceOp(SSAGraph* graph) const;

  void FuseAllReduceOp(
      SSAGraph* graph,
      std::unordered_set<std::unique_ptr<OpHandleBase>>&& ops) const;

  std::unique_ptr<SSAGraphBuilder> builder_;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
