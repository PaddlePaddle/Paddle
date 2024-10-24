// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>

#include "glog/logging.h"

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/ir/xpu/pass_utils.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

struct StackPattern : public PatternBase {
  StackPattern(PDPattern* pattern, const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(stack);
  // declare variable node's name
  PATTERN_DECL_NODE(stack_out);
};

StackPattern::StackPattern(PDPattern* pattern, const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* stack = pattern->NewNode(stack_repr())
                    ->assert_is_op("stack")
                    ->assert_more([](Node* node) {
                      auto input_names = node->Op()->Input("X");
                      auto first_name = input_names[0];
                      for (auto name : input_names) {
                        if (name != first_name) return false;
                      }
                      return true;
                    });
  auto* stack_out = pattern->NewNode(stack_out_repr())
                        ->assert_is_op_output("stack", "Y")
                        ->assert_more([](Node* node) {
                          std::map<std::string, std::string> support_out_ops{
                              {"elementwise_add", "Y"},
                              {"fused_multi_transformer", "SrcMask"}};
                          auto var_name = node->Name();
                          for (auto* out_node : node->outputs) {
                            auto op_type = out_node->Op()->Type();
                            if (support_out_ops.count(op_type) == 0)
                              return false;
                            auto out_op_in_names =
                                out_node->Op()->Input(support_out_ops[op_type]);
                            if (std::find(out_op_in_names.begin(),
                                          out_op_in_names.end(),
                                          var_name) == out_op_in_names.end())
                              return false;
                          }
                          return true;
                        });
  stack->LinksTo({stack_out});
}

}  // namespace patterns

/*
"stack" can be replaced by "unsqueeze" if:
1. "stack inputs" are the same。
2. "stack output" is "elementwise_add input" or "fused_multi_transformer
src_mask input".
*/
class StackFusePass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  const std::string name_scope_{"stack_fuse_pass"};
};

void StackFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);
  GraphPatternDetector gpd;
  patterns::StackPattern pattern(gpd.mutable_pattern(), name_scope_);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle StackFusePass fuse";
    GET_IR_NODE(stack);
    GET_IR_NODE(stack_out);

    stack->RenameOp("unsqueeze2");
    auto* op_desc = stack->Op();
    int axis = op_desc->GetAttrIfExists<int>("axis");
    op_desc->SetAttr("axes", std::vector<int>{axis});
    op_desc->RemoveAttr("axis");

    op_desc->MutableInputs()->at("X").resize(1);
    auto* stack_in = stack->inputs[0];
    IR_NODE_UNLINK(stack_in, stack);
    IR_NODE_LINK_TO(stack_in, stack);

    auto* outputs = op_desc->MutableOutputs();
    (*outputs)["Out"] = outputs->at("Y");
    outputs->erase("Y");

    auto stack_out_shape = stack_out->Var()->GetShape();
    if (axis < 0) axis += stack_out_shape.size();
    stack_out_shape[axis] = 1;
    stack_out->Var()->SetShape(stack_out_shape);

    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(stack_fuse_pass, paddle::framework::ir::StackFusePass);

REGISTER_PASS_CAPABILITY(stack_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "stack", 0));
