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

class FuseAdjacentNodesPass : public Pass {
 protected:
  std::unique_ptr<ir::Graph> ApplyImpl(
      std::unique_ptr<ir::Graph> graph) const override;

 private:
  /**
   * Whether the two nodes can be fused.
   * upstream_op_node's output is the input of cur_op_node here.
   */
  bool IsFusible(const NodePtr cur_op_node,
                 const NodePtr upstream_op_node) const;

  /**
   * Whether cur_op_node and some of it's adjacent nodes can be fused.
   * If the return value is true, the cur_op_node can be fused with
   * its adjacent nodes and the adjacent nodes is stored in tobe_fused_nodes.
   * If cur_op_node has been fused, cur_op_node will be reset to the fused node
   * which is stored in m_internal.
   */
  bool FindToBeFusedNodes(
      const NodePtr cur_op_node,
      const std::unordered_map<NodePtr, InternalNodePtr> &m_internal,
      std::unordered_set<NodePtr> *tobe_fused_nodes) const;

  /**
   * Fuse cur_op_node and tobe_fused_nodes, and insert the fused node graph.
   * In this process, some nodes will become useless, and they are stored in
   * need_removed_nodes.
   */
  NodePtr FuseNodes(const NodePtr cur_op_node,
                    const std::unordered_set<NodePtr> &tobe_fused_nodes,
                    std::unordered_set<NodePtr> *need_removed_nodes,
                    ir::Graph *graph) const;

  /**
   *  Generate the op_desc of the fused op.
   */
  void FuseElemwiseAndActivation(
      const NodePtr node, const std::unordered_set<NodePtr> &tobe_fused,
      OpDesc *op_desc, std::unordered_set<NodePtr> *intermediate_out) const;

  /**
   *  The node and tobe_fused_nodes should be in the same stage,
   *  and the stage can be forward, backward and parameter optimization.
   */
  bool IsBackward(const NodePtr node,
                  const std::unordered_set<NodePtr> &tobe_fused_nodes) const;

  bool IsElemwiseAndActivation(
      const NodePtr node,
      const std::unordered_set<NodePtr> &tobe_fused_node) const;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
