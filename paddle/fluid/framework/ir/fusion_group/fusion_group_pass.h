/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include <unordered_set>

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/fusion_group/subgraph.h"

namespace paddle {
namespace framework {
namespace ir {

class Graph;

namespace fusion_group {
class SubGraph;
}  // namespace fusion_group

class FusionGroupPass : public FusePassBase {
 protected:
  void ApplyImpl(Graph* graph) const override;

 private:
  int DetectFusionGroup(Graph* graph, int type = 0) const;
  bool GenerateCode(fusion_group::SubGraph* subgraph) const;
  void InsertFusionGroupOp(Graph* graph,
                           fusion_group::SubGraph* subgraph) const;

  const std::string name_scope_{"fusion_group"};
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
