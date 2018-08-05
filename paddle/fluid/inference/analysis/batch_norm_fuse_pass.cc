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

#include "paddle/fluid/inference/analysis/fuse.h"
#include "paddle/fluid/inference/analysis/pass.h"

namespace paddle {
namespace inference {
namespace analysis {

/*
 * The batch normalization followed the convolution or fully connected
 * layer
 *        can be integrated with them. Doing so will give us a forward
 * acceleration,
 *       especially in environments like mobile or embedded.
 *       For input :math:`X`:
 *       - Conv process:        :math:`X = input * W + bias`
 *       - Batch norm process:  :math:`X' = (X - mean) / std`
 *       - Scale Process:       :math:`Y = a * X' + b`
 *       After fuse into one operation:
 *       .. math::
 *           Y &= (input * W + bias - mean) / std * a + b \\\\
 *             &= input * a * W / std + ((bias - mean) / std * a + b)
 *       The operator transformation is:
 *       - before:
 *         - conv->batch_norm->any_other_op (bias == 0)
 *         - conv->elementwise_add->batch_norm->any_other_op (bias != 0)
 *       - after:
 *         - conv->elementwise_add->any_other_op
 */
class BatchNormFusePass : public DataFlowGraphPass {
 public:
  virtual bool Initialize(Argument *argument) {
    PADDLE_ENFORCE(argument, "invalid argument");
    argument_ = argument;

#define __add_function__(op_type__) \
  auto *op_type__##_pnode = AddPatternNode(#op_type__);
#define __add_value__(op_type__) \
  auto *op_type__##_out_pnode = AddOutputPatternNode(#op_type__);
    // A topological pattern:
    // conv2d -> var -> batch_norm -> var -> element_wise_add
    __add_function__(conv2d);
    __add_value__(conv2d);
    __add_function__(batch_norm);
    __add_value__(batch_norm);
    __add_function__(element_wise_add);
#undef __add_function__
#undef __add_value__

    fuse_tactic_.AddEdge(conv2d_pnode, conv2d_out_pnode);
    fuse_tactic_.AddEdge(conv2d_out_pnode, batch_norm_pnode);
    fuse_tactic_.AddEdge(batch_norm_pnode, batch_norm_out_pnode);
    fuse_tactic_.AddEdge(batch_norm_out_pnode, element_wise_add_pnode);

    // Remove other nodes, and insert a new node.
    fuse_tactic_.SetHandle(
        [](const fuse::PatternRecord &pattern, DataFlowGraph *graph) {
          for (auto &item : pattern.symbol_table) {
            item.second->SetDeleted();  // set a mark
          }
          // TODO(Superjomn) Insert a new Function Node, should be much comple.
        });

    return true;
  }

  virtual void Run(DataFlowGraph *graph) { fuse_tactic_.Fuse(graph); }

 private:
  // Add a pattern node to fuse tactic
  FusePatternNode *AddPatternNode(const std::string &op_type) {
    auto *pnode = fuse_tactic_.AddNode();
    pnode->teller = [op_type](Node *node) {
      return node->IsFunction() &&
             dynamic_cast<Function *>(node)->func_type() == op_type;
    };
    return pnode;
  }

  /*
   * Add an Value Node which is a `op_type` Function's output.
   * Free to add other checks.
   */
  FusePatternNode *AddOutputPatternNode(const std::string &op_type) {
    auto *pnode = fuse_tactic_.AddNode();
    pnode->teller = [op_type](Node *node) {
      if (!node->IsValue()) return false;
      // Check whether this node is an output of conv2d
      for (auto *func : dynamic_cast<Value *>(node)->inlinks) {
        if (func->IsFunction() &&
            dynamic_cast<Function *>(func)->func_type() == op_type) {
          return true;
        }
      }
      return false;
    };
    return pnode;
  }

 private:
  Argument *argument_{nullptr};
  fuse::Pattern fuse_tactic_;
};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
