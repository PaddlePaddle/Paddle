/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/constant_folding_pass.h"
#include <string>
#include <vector>
#include "glog/logging.h"
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

struct ConstantFolding : public PatternBase {
  ConstantFolding(PDPattern *pattern, const std::string &name_scope)
      : PatternBase(pattern, name_scope, "constant_folding_pass") {}

  PDNode *operator()(PDNode *x);

  // declare operator node's name
  PATTERN_DECL_NODE(common_op);
  // declare variable node's name
  // common_op_in may be null, so ommit it
  PATTERN_DECL_NODE(common_op_out);
};

PDNode *ConstantFolding::operator()(PDNode *persis_op) {
  auto assert_ops = std::unordered_set<std::string>{
      "unsqueeze2", "reshape2", "fill_constant"};
  persis_op->assert_is_ops(assert_ops);
  auto *op_out = pattern->NewNode(common_op_out_repr())
                     ->assert_is_ops_output(assert_ops, "Out")
                     ->assert_more([&](Node *node) {
                       auto next_ops = std::unordered_set<std::string>{
                           "set_value", "conditional_block", "while"};
                       for (size_t i = 0; i < node->outputs.size(); i++) {
                         auto op_type = node->outputs[i]->Op()->Type();
                         if (next_ops.count(op_type)) return false;
                       }
                       return true;
                     });

  op_out->LinksFrom({persis_op});
  return op_out;
}

}  // namespace patterns

ConstantFoldingPass::ConstantFoldingPass() {}

static bool ValidateOp(Node *op) {
  if (op->inputs.size() <= 1) {
    return true;
  } else if (op->inputs.size() == 2) {
    for (int i = 0; i < 2; i++) {
      auto input_i_persis = op->inputs[i]->Var()->Persistable();
      auto input_i_outnum = op->inputs[i]->outputs.size();
      if (input_i_persis && input_i_outnum == 1) return true;
    }
    return false;
  } else {
    return false;
  }
}

void ConstantFoldingPass::ApplyImpl(ir::Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  FusePassBase::Init("constant_folding", graph);
  int found_subgraph_count = 0;
  GraphPatternDetector gpd;
  auto *persis_op_node = gpd.mutable_pattern()
                             ->NewNode("persis_op_node")
                             ->assert_is_op()
                             ->assert_more([&](Node *node) {
                               int op_input_size = node->inputs.size();
                               if (op_input_size == 1) {
                                 auto *input_node = node->inputs[0];
                                 return input_node->Var()->Persistable() &&
                                        input_node->outputs.size() == 1;
                               } else if (op_input_size == 0) {
                                 return true;
                               } else {
                                 return false;
                               }
                             });

  patterns::ConstantFolding fused_pattern(gpd.mutable_pattern(),
                                          "constant_folding");
  fused_pattern(persis_op_node);

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *graph) {
    if (subgraph.count(persis_op_node) <= 0) {
      LOG(WARNING) << "The subgraph is empty.";
      return;
    }
    VLOG(4) << "handle ConstantFolding pass";

    GET_IR_NODE_FROM_SUBGRAPH(op_out, common_op_out, fused_pattern);

    auto *scope = param_scope();
    framework::Scope *new_scope = new framework::Scope();
    std::unordered_set<const paddle::framework::ir::Node *> remove_nodes;

    if (subgraph.at(persis_op_node)->inputs.size()) {
      auto *persis_x_node = subgraph.at(persis_op_node)->inputs[0];
      auto x_shape = persis_x_node->Var()->GetShape();
      auto *persis_x_tensor =
          scope->FindVar(persis_x_node->Name())->GetMutable<LoDTensor>();
      new_scope->Var(persis_x_node->Name());
      auto *x_tensor =
          new_scope->FindVar(persis_x_node->Name())->GetMutable<LoDTensor>();
      x_tensor->Resize(persis_x_tensor->dims());
      *x_tensor = *persis_x_tensor;
    }

    auto *iter_op = subgraph.at(persis_op_node);
    std::vector<std::unique_ptr<OperatorBase>> ops;

    // for Op Node, it return the op's ith output, and make sure all others
    // outputs muse be useless for Var Node, it return 0th output if
    // outputs.size() == 1 and then return nullptr
    auto PickOneOut = [&](Node *node) -> Node * {
      if (node->outputs.size() == 1) {
        return node->outputs[0];
      } else if (node->outputs.size() == 0) {
        return nullptr;
      } else {
        if (node->IsOp()) {
          // Pick out the most useful output var
          int useful_out_num = 0;
          int useful_out_index = -1;
          for (size_t i = 0; i < node->outputs.size(); i++) {
            if (node->outputs[i]->outputs.size() >= 1) {
              useful_out_num++;
              useful_out_index = i;
            }
          }
          if (useful_out_num == 1) return node->outputs[useful_out_index];
        } else {
          return nullptr;
        }
      }
      return nullptr;
    };

    auto *last_persis_node = op_out;
    while (iter_op) {
      if (ValidateOp(iter_op)) {
        remove_nodes.emplace(iter_op);
        for (auto in_node : iter_op->inputs) {
          new_scope->Var(in_node->Var()->Name());
          auto in_node_tensor = new_scope->FindVar(in_node->Var()->Name())
                                    ->GetMutable<LoDTensor>();
          if (in_node->Var()->Persistable()) {
            *in_node_tensor = *(scope->FindVar(in_node->Var()->Name())
                                    ->GetMutable<LoDTensor>());
          }
          remove_nodes.emplace(in_node);
        }
        for (auto out_node : iter_op->outputs) {
          new_scope->Var(out_node->Var()->Name());
          new_scope->FindVar(out_node->Var()->Name())->GetMutable<LoDTensor>();
        }
        ops.emplace_back(
            paddle::framework::OpRegistry::CreateOp(*iter_op->Op()));
        auto out_node = PickOneOut(iter_op);
        last_persis_node = out_node;
        iter_op = PickOneOut(out_node);
      } else {
        break;
      }
    }

    for (size_t i = 0; i < ops.size(); i++) {
      ops[i]->Run(*new_scope, platform::CPUPlace());
    }

    auto out_desc = last_persis_node->Var();
    auto out_name = out_desc->Name();
    auto *local_out_tensor =
        new_scope->FindVar(out_name)->GetMutable<LoDTensor>();

    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < local_out_tensor->dims().size(); i++) {
      out_shape.push_back(local_out_tensor->dims()[i]);
    }
    out_desc->SetShape(out_shape);
    out_desc->SetPersistable(true);
    auto *out_tensor = scope->Var(out_name)->GetMutable<LoDTensor>();
    *out_tensor = *local_out_tensor;

    delete new_scope;
    // Remove links in graph
    GraphSafeRemoveNodes(graph, remove_nodes);
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(constant_folding_pass,
              paddle::framework::ir::ConstantFoldingPass);
