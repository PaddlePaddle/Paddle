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
#include <string>
#include <typeindex>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/details/fuse_vars_op_handle.h"
#include "paddle/fluid/framework/details/nccl_all_reduce_op_handle.h"
#include "paddle/fluid/framework/details/ssa_graph_builder.h"

namespace paddle {
namespace framework {
namespace details {

class FuseAllReduceGraphBuilder : public SSAGraphBuilder {
 public:
  explicit FuseAllReduceGraphBuilder(std::unique_ptr<SSAGraphBuilder> &&builder,
                                     const std::vector<platform::Place> &places,
                                     const std::vector<Scope *> &local_scopes,
                                     platform::NCCLContextMap *ctxs)
      : builder_(std::move(builder)),
        places_(places),
        local_scopes_(local_scopes),
        ctxs_(ctxs) {}

  std::unique_ptr<SSAGraph> Build(const ProgramDesc &program) const override;

 private:
  std::unique_ptr<SSAGraphBuilder> builder_;
  const std::vector<platform::Place> places_;
  const std::vector<Scope *> local_scopes_;
  platform::NCCLContextMap *ctxs_;

 private:
  struct NCCLAllReduceGroup {
    std::unordered_set<std::unique_ptr<OpHandleBase>> ops_;
    std::type_index type_{typeid(void)};
  };
  /**
   * Get All-Reduce operator into multiple sets.
   * The order of set is the order of execution.
   */
  std::vector<NCCLAllReduceGroup> GetNotDependedAllReduceOp(
      SSAGraph *graph, const BlockDesc &global_block) const;

  void FuseAllReduceOp(SSAGraph *graph, NCCLAllReduceGroup &&ops,
                       const BlockDesc &global_block) const;

  std::vector<VarHandle *> GetFusedGradient(
      SSAGraph *graph, const BlockDesc &global_block,
      const NCCLAllReduceGroup &ops) const;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
