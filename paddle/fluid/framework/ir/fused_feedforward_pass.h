// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <mutex>
#include <string>
#include <unordered_set>

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {

/*
 * Fuse the FeedForward in attention
 * Forward:
 *   1. layer_norm -> linear1 -> activation -> dropout1 -> linear2 -> dropout2
 * -> residual_add (pre_layer_norm)
 *   2. linear1 -> activation -> dropout1 -> linear2 -> dropout2 -> residual_add
 * -> layer_norm (pose_layer_norm)
 *   other cases: may delete mp, residual_add, dropout1, dropout2 operators
 * Backward:
 *   1. residual_add_grad -> dropout2_grad -> linear2_grad -> dropout1_grad ->
 * activation_grad -> linear1_grad -> layer_norm_grad (pre_layer_norm)
 *   2. layer_norm_grad -> residual_add_grad -> dropout2_grad -> linear2_grad ->
 * dropout1_grad -> activation_grad -> linear1_grad (pose_layer_norm)
 *   other cases: may delete mp, residual_add_grad, dropout1_grad, dropout2_grad
 * operators
 */
class Graph;
class Node;

class FusedFeedForwardPass : public FusePassBase {
 public:
  virtual ~FusedFeedForwardPass() {}

 protected:
  // Used for pattern created variable node transfer
  // between corresponding forward operator and backward operator.
  struct DropoutNode {
    Node *dropout_out_node_1;
    Node *dropout_mask_node_1;
    Node *dropout_out_node_2;
    Node *dropout_mask_node_2;
    DropoutNode()
        : dropout_out_node_1(nullptr),
          dropout_mask_node_1(nullptr),
          dropout_out_node_2(nullptr),
          dropout_mask_node_2(nullptr) {}
  };
  typedef std::unordered_map<Node *, DropoutNode> Cache;

  const std::string scope_name{"fused_feedforward"};

  void ApplyImpl(ir::Graph *graph) const override;

  ir::Graph *FusedFeedForwardFwd(ir::Graph *graph,
                                 bool use_mp,
                                 bool pre_layer_norm,
                                 bool add_residual,
                                 bool use_dropout_1,
                                 bool use_dropout_2,
                                 Cache *dropout_nodes_map) const;

  ir::Graph *FusedFeedForwardBwd(ir::Graph *graph,
                                 bool use_mp,
                                 bool pre_layer_norm,
                                 bool add_residual,
                                 bool use_dropout_1,
                                 bool use_dropout_2,
                                 Cache *dropout_nodes_map) const;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
