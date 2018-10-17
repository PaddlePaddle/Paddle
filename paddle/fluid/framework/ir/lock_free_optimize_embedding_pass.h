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

#ifndef PADDLE_FLUID_FRAMEWORK_IR_LOCK_FREE_OPTIMIZE_EMBEDDING_PASS_H_
#define PADDLE_FLUID_FRAMEWORK_IR_LOCK_FREE_OPTIMIZE_EMBEDDING_PASS_H_

#include <string>

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {

class Node;

/*
 * Remove the sum op of all gradients of embedding lookup table.
 * And remove the dependecies of the optimizer related to the
 * same embedding lookup table.
 */
class LockFreeOptimizeEmbeddingPass : public Pass {
 public:
  virtual ~LockFreeOptimizeEmbeddingPass() {}

 protected:
  std::unique_ptr<ir::Graph> ApplyImpl(std::unique_ptr<ir::Graph> graph) const;

 private:
  // Create a new optimizer node via current optimizer node
  ir::Node* CreateNewOptimizerNode(ir::Graph* graph, ir::Node* grad_node,
                                   ir::Node* grad_output_node,
                                   const std::string& optimizer_type,
                                   const std::string& grad_name) const;

  // Replace the input weight's optimizers
  void ReplaceUpstreamOptimizerNode(ir::Node* upstream_node,
                                    ir::Node* old_optimizer_node,
                                    ir::Node* new_optimizer_node) const;

  // Replace the output weight's optimizers
  void ReplaceDownstreamOptimizerNode(ir::Node* downstream_node,
                                      ir::Node* old_optimizer_node,
                                      ir::Node* new_optimizer_node) const;

  // Check if ctrl_dep_var_node's input op equals to
  // the lookup_table_grad_output_node's related lookup_table op
  bool IsRelatedEmbeddingOp(ir::Node* ctrl_dep_var_node,
                            ir::Node* lookup_table_grad_output_node) const;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle

#endif  // PADDLE_FLUID_FRAMEWORK_IR_LOCK_FREE_OPTIMIZE_EMBEDDING_PASS_H_
