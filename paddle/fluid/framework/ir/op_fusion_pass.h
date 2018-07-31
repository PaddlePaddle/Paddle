/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {

using NodePtr = Node *;
using InternalNodePtr = Node *;

class OpFusionPass : public Pass {
 protected:
  std::unique_ptr<ir::Graph> ApplyImpl(
      std::unique_ptr<ir::Graph> graph) const override;

 private:
  bool IsFusible(const NodePtr n1, const NodePtr n2) const;

  bool SetUpFusion(
      const NodePtr node,
      const std::unordered_map<NodePtr, InternalNodePtr> &m_internal,
      std::unordered_set<NodePtr> *tobe_fused) const;

  NodePtr FuseOperators(const NodePtr cur_node,
                        const std::unordered_set<NodePtr> &tobe_fused,
                        std::unordered_set<NodePtr> *need_removed_node,
                        ir::Graph *graph) const;

  bool IsBackward(const NodePtr node,
                  const std::unordered_set<NodePtr> &tobe_fused) const;

  bool IsElemwiseAndActivation(
      const NodePtr node, const std::unordered_set<NodePtr> &tobe_fused) const;

  void FuseElemwiseAndActivation(const NodePtr node,
                                 const std::unordered_set<NodePtr> &tobe_fused,
                                 OpDesc *op_desc) const;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
