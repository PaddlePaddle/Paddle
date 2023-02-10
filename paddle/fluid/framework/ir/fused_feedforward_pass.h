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
 * -> residual_add
 *   2. layer_norm -> linear1 -> activation -> dropout1 -> linear2 -> dropout2
 *   3. linear1 -> activation -> dropout1 -> linear2 -> dropout2 -> residual_add
 * -> layer_norm
 *   4. linear1 -> activation -> dropout1 -> linear2 -> dropout2 -> layer_norm
 * Backward:
 *   1. residual_add_grad -> dropout2_grad -> linear2_grad -> dropout1_grad ->
 * activation_grad -> linear1_grad -> layer_norm_grad
 *   2. dropout2_grad -> linear2_grad -> dropout1_grad -> activation_grad ->
 * linear1_grad -> layer_norm_grad
 *   3. layer_norm_grad -> residual_add_grad -> dropout2_grad -> linear2_grad ->
 * dropout1_grad -> activation_grad -> linear1_grad
 *   4. layer_norm_grad -> dropout2_grad -> linear2_grad -> dropout1_grad ->
 * activation_grad -> linear1_grad
 */
class Graph;
class Node;

class FusedFeedForwardPass : public FusePassBase {
 public:
  virtual ~FusedFeedForwardPass() {}

 protected:
  const std::string scope_name{"fused_feedforward"};

  void ApplyImpl(ir::Graph *graph) const override;

  ir::Graph *FusedFeedForwardFwd(ir::Graph *graph,
                                 bool pre_layer_norm,
                                 bool add_residual) const;

  ir::Graph *FusedFeedForwardBwd(ir::Graph *graph,
                                 bool pre_layer_norm,
                                 bool add_residual) const;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
