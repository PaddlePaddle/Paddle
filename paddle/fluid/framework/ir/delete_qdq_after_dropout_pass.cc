/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/delete_qdq_after_dropout_pass.h"
#include <string>
#include <vector>
#include "glog/logging.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

#include "paddle/fluid/framework/convert_utils.h"

namespace paddle {
namespace framework {
namespace ir {
class Node;
}  // namespace ir
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

struct DeleteQdqAfterDropoutPass : public PatternBase {
  DeleteQdqAfterDropoutPass(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "delete_qdq_after_dropout_pass") {}
};
}  // namespace patterns

DeleteQdqAfterDropoutPass::DeleteQdqAfterDropoutPass() {}

void DeleteQdqAfterDropoutPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  FusePassBase::Init("delete_qdq_after_dropout_pass", graph);
  auto* scope = param_scope();

  PADDLE_ENFORCE_NOT_NULL(
      scope,
      platform::errors::Fatal("scope must not be null when applying "
                              "delete_qdq_after_dropout_pass."));
  GraphPatternDetector detector;
  auto dropout_op =
      detector.mutable_pattern()->NewNode("dropout")->assert_is_op("dropout");
  auto dropout_op_out = detector.mutable_pattern()
                            ->NewNode("dropout_op_out")
                            ->assert_is_op_output("dropout");
  auto q_op = detector.mutable_pattern()->NewNode("q_op")->assert_is_op(
      "quantize_linear");
  auto q_op_out =
      detector.mutable_pattern()
          ->NewNode("q_op_out")
          ->assert_is_op_output("quantize_linear")
          ->assert_more([](Node* x) { return x->outputs.size() == 1UL; });
  auto dq_op = detector.mutable_pattern()->NewNode("dq_op")->assert_is_op(
      "dequantize_linear");
  auto dq_op_out = detector.mutable_pattern()
                       ->NewNode("dq_op_out")
                       ->assert_is_op_output("dequantize_linear");

  dropout_op->LinksTo({dropout_op_out});
  dropout_op_out->LinksTo({q_op});
  q_op->LinksTo({q_op_out});
  q_op_out->LinksTo({dq_op});
  dq_op->LinksTo({dq_op_out});
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    // Node* dropout_op_node = subgraph.at(dropout_op);
    Node* dropout_op_out_node = subgraph.at(dropout_op_out);
    Node* q_op_node = subgraph.at(q_op);
    Node* q_op_out_node = subgraph.at(q_op_out);
    Node* dq_op_node = subgraph.at(dq_op);
    Node* dq_op_out_node = subgraph.at(dq_op_out);

    // dropout_op_node
    //       |
    // dropout_op_out_node
    //    |         |    |
    // q_op_node  others others
    //    |
    // q_op_out_node
    //     |
    // dq_op_node
    //    |
    // dq_op_out_node
    //  |    |    |
    // op1  op2  op3

    std::cout << dropout_op_out_node->Name() << std::endl;
    std::vector<Node*> new_dropout_op_outputs;
    for (auto next_op : dropout_op_out_node->outputs) {
      if (next_op != q_op_node) {
        new_dropout_op_outputs.push_back(next_op);
      }
    }
    for (auto& next_op : dq_op_out_node->outputs) {
      new_dropout_op_outputs.push_back(next_op);
    }
    dropout_op_out_node->outputs = new_dropout_op_outputs;

    for (auto next_op : dq_op_out_node->outputs) {
      std::vector<Node*> new_op_inputs;
      for (auto& in : next_op->inputs) {
        if (in != dq_op_out_node) {
          new_op_inputs.push_back(in);
        }
      }
      new_op_inputs.push_back(dropout_op_out_node);
      next_op->inputs = new_op_inputs;
      auto op_desc = next_op->Op();
      op_desc->RenameInput(dq_op_out_node->Var()->Name(),
                           dropout_op_out_node->Var()->Name());
      op_desc->Flush();
    }

    GraphSafeRemoveNodes(graph,
                         {
                             q_op_node,
                             q_op_out_node,
                             dq_op_node,
                             dq_op_out_node,
                         });
  };
  if (0) detector(graph, handler);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(delete_qdq_after_dropout_pass,
              paddle::framework::ir::DeleteQdqAfterDropoutPass);
