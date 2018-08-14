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
   * Whether cur_op_node and upstream_op_node can be fused.
   * cur_op_node and upstream_op_node should be Operation and upstream_op_node's
   * output is the input of cur_op_node here.
   *
   * The condition of fusing cur_op_node and upstream_op_node is:
   *   - the number of upstream_op_node's outputs(not include ControlDepVar)
   *     should be one.
   *   - upstream_op_node's output is only used by cur_op_node or
   *     cur_op_grad_node or upstream_op_grad_node.
   *   - there is a template function to represent the fused of cur_op_node and
   * upstream_op_node.
   */
  bool IsFusible(const NodePtr cur_op_node,
                 const NodePtr upstream_op_node) const;

  /**
   * Whether cur_op_node and some of it's adjacent nodes can be fused.
   * If the return value is true, the cur_op_node and tobe_fused_nodes can be
   * fused.
   * If cur_op_node has been fused, cur_op_node will be reset to the fused node
   * which is stored
   * in m_internal.
   */
  bool FindToBeFusedNodes(
      const NodePtr cur_op_node,
      const std::unordered_map<NodePtr, InternalNodePtr> &m_internal,
      std::unordered_set<NodePtr> *tobe_fused_nodes) const;

  /**
   * Fuse cur_op_node and the nodes of tobe_fused_nodes.
   * Insert the fused node graph.
   * Collection of nodes that are no longer useful.
   */
  NodePtr FuseNodes(const NodePtr cur_op_node,
                    const std::unordered_set<NodePtr> &tobe_fused_nodes,
                    std::unordered_set<NodePtr> *need_removed_nodes,
                    ir::Graph *graph) const;

  void FuseElemwiseAndActivation(const NodePtr node,
                                 const std::unordered_set<NodePtr> &tobe_fused,
                                 OpDesc *op_desc) const;

  bool IsBackward(const NodePtr node,
                  const std::unordered_set<NodePtr> &tobe_fused) const;

  bool IsElemwiseAndActivation(
      const NodePtr node, const std::unordered_set<NodePtr> &tobe_fused) const;

  void AddAbsentNodes(const NodePtr cur_op_node,
                      const std::unordered_set<NodePtr> &tobe_fused_nodes,
                      Node *fused_node) const;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
