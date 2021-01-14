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

#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {

/*
 * Fuse the BatchNorm and activation.
 */
class Graph;
class Node;

class FuseBatchNormActPass : public FusePassBase {
 public:
  virtual ~FuseBatchNormActPass() {}

 protected:
  void ApplyImpl(ir::Graph *graph) const override;

  ir::Graph *FuseBatchNormAct(
      ir::Graph *graph, const std::unordered_set<std::string> &act_types) const;

  ir::Graph *FuseBatchNormActGrad(
      ir::Graph *graph,
      const std::unordered_set<std::string> &act_grad_types) const;

  std::vector<Node *> ReplaceNode(Node *cur_node, Node *new_node,
                                  const std::vector<Node *> &nodes) const;

  void ReLinkNodes(Graph *graph, const Node *intermediate_out, Node *op_1,
                   Node *op_2, Node *fused_op) const;
  Node *CreateFusedBatchNormActNode(
      Graph *g, const Node *act, const Node *bn, const std::string &bn_x_n,
      const std::string &bn_scale_n, const std::string &bn_bias_n,
      const std::string &bn_variance_n, const std::string &bn_mean_n,
      const std::string &bn_mean_out_n, const std::string &bn_variance_out_n,
      const std::string &bn_saved_variance_n,
      const std::string &bn_saved_mean_n, const std::string &bn_reserve_space_n,
      const std::string &act_out_n) const;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
