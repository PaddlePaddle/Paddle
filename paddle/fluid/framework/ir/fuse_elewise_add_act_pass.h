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
 * Fuse the ElewiseAdd and activation
 */
class Graph;
class Node;

class FuseElewiseAddActPass : public FusePassBase {
 public:
  virtual ~FuseElewiseAddActPass() {}

 protected:
  void ApplyImpl(ir::Graph *graph) const override;

  ir::Graph *FuseElewiseAddAct(
      ir::Graph *graph, const std::unordered_set<std::string> &act_types) const;

  ir::Graph *FuseActElewiseAdd(
      ir::Graph *graph, const std::unordered_set<std::string> &act_types) const;

  ir::Graph *FuseElewiseAddActInplaceGrad(
      ir::Graph *graph, const std::unordered_set<std::string> &act_types) const;

  /**
   * Remove the removable intermediate_out.
   *   - If the intermediate_out is only used by the backward op, but the
   *     backward op doesn't use intermediate_out.
   *   - If the intermediate_out_grad is not used by any op.
   */
  void RemoveIntermediateOut(Graph *graph) const;

  std::vector<Node *> ReplaceNode(Node *cur_node, Node *new_node,
                                  const std::vector<Node *> &nodes) const;

  std::vector<Node *> RemoveNode(Node *trg_node,
                                 const std::vector<Node *> &nodes) const;

  void ReLinkNodes(Graph *graph, const Node *intermediate_out, Node *op_1,
                   Node *op_2, Node *fused_op) const;
  Node *CreateFuseElewiseAddActNode(Graph *g, const Node *op_1,
                                    const Node *op_2,
                                    const std::string &ele_x_n,
                                    const std::string &ele_y_n,
                                    const std::string &ele_out_n,
                                    const std::string &act_out_n) const;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
