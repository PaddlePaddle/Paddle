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

void ConstantFoldingPass::ApplyImpl(ir::Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  FusePassBase::Init("constant_folding", graph);
  auto *scope = param_scope();

  auto op_node_sorted = framework::ir::TopologyVarientSort(
      *graph, static_cast<framework::ir::SortKind>(0));
  for (auto *op_node : op_node_sorted) {
    if (!op_node->IsOp()) continue;
    if (op_node->Op()->Type() == "feed") continue;
    bool input_persis = true;
    for (auto in_node : op_node->inputs) {
      if (!in_node->infered_persistable && !in_node->Var()->Persistable()) {
        input_persis = false;
      }
    }
    framework::Scope *new_scope = new framework::Scope();
    std::unordered_set<const paddle::framework::ir::Node *> remove_nodes;
    std::unique_ptr<OperatorBase> op;

    for (auto out_node : op_node->outputs) {
      for (auto next_op : out_node->outputs) {
        if (next_op->Op()->Type() == "set_value") {
          input_persis = false;
        }
      }
    }

    if (input_persis) {
      for (auto in_node : op_node->inputs) {
        new_scope->Var(in_node->Var()->Name());
        new_scope->FindVar(in_node->Var()->Name())->GetMutable<LoDTensor>();
        // this input persistable is exclusive
        if (in_node->outputs.size() == 1L) remove_nodes.emplace(in_node);

        auto in_shape = in_node->Var()->GetShape();
        auto *persis_x_tensor =
            scope->FindVar(in_node->Name())->GetMutable<LoDTensor>();
        auto *x_tensor =
            new_scope->FindVar(in_node->Name())->GetMutable<LoDTensor>();
        x_tensor->Resize(persis_x_tensor->dims());
        *x_tensor = *persis_x_tensor;
      }

      op = paddle::framework::OpRegistry::CreateOp(*op_node->Op());
      remove_nodes.emplace(op_node);
      for (auto out_node : op_node->outputs) {
        out_node->infered_persistable = true;
        new_scope->Var(out_node->Var()->Name());
        new_scope->FindVar(out_node->Var()->Name())->GetMutable<LoDTensor>();
        if (out_node->outputs.size() == 0L) remove_nodes.emplace(out_node);
        std::cout << out_node->Var()->Name() << std::endl;
        std::cout << out_node->outputs.size() << std::endl;
      }
      op->Run(*new_scope, platform::CPUPlace());
      for (auto out_node : op_node->outputs) {
        if (out_node->outputs.size() == 0L) continue;
        auto out_desc = out_node->Var();
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
      }
      GraphSafeRemoveNodes(graph, remove_nodes);
    }
    delete new_scope;
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(constant_folding_pass,
              paddle::framework::ir::ConstantFoldingPass);
