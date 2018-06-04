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

#include "paddle/fluid/framework/details/fuse_all_reduce_graph_builder.h"
#include <unordered_set>
#include <vector>

namespace paddle {
namespace framework {
namespace details {
std::unique_ptr<SSAGraph> FuseAllReduceGraphBuilder::Build(
    const ProgramDesc &program) const {
  // TODO(yy): Complete this method.
  auto graph = builder_->Build(program);

  auto all_reduce_ops = GetNotDependedAllReduceOp(graph.get());

  for (auto &op_group : all_reduce_ops) {
    FuseAllReduceOp(graph.get(), std::move(op_group));
  }
  return graph;
}
std::vector<std::unordered_set<std::unique_ptr<OpHandleBase>>>
FuseAllReduceGraphBuilder::GetNotDependedAllReduceOp(SSAGraph *graph) const {
  return std::vector<std::unordered_set<std::unique_ptr<OpHandleBase>>>();
}
void FuseAllReduceGraphBuilder::FuseAllReduceOp(
    SSAGraph *graph,
    std::unordered_set<std::unique_ptr<OpHandleBase>> &&ops) const {}
}  // namespace details
}  // namespace framework
}  // namespace paddle
