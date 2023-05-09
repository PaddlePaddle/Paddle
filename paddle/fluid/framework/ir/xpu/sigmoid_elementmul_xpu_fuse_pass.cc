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
#include "paddle/fluid/framework/ir/xpu/quant_utils.h"
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

struct SwishXPUPattern : public PatternBase {
  SwishXPUPattern(PDPattern* pattern, const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(sigmoid);
  PATTERN_DECL_NODE(elementwise_mul);
  // declare variable node's name
  PATTERN_DECL_NODE(sigmoid_x);
  PATTERN_DECL_NODE(sigmoid_out);
  PATTERN_DECL_NODE(elemul_out);
};

SwishXPUPattern::SwishXPUPattern(PDPattern* pattern,
                                 const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* sigmoid_x = pattern->NewNode(sigmoid_x_repr())
                        ->assert_is_op_input("sigmoid", "X")
                        ->assert_var_not_persistable();

  auto* sigmoid_op = pattern->NewNode(sigmoid_repr())->assert_is_op("sigmoid");

  auto* sigmoid_out = pattern->NewNode(sigmoid_out_repr())
                          ->assert_is_op_output("sigmoid", "Out")
                          ->assert_var_not_persistable();

  auto* elemul_op =
      pattern->NewNode(elementwise_mul_repr())->assert_is_op("elementwise_mul");

  auto* elemul_out = pattern->NewNode(elemul_out_repr())
                         ->assert_is_op_output("elementwise_mul", "Out")
                         ->assert_var_not_persistable();

  sigmoid_op->LinksFrom({sigmoid_x}).LinksTo({sigmoid_out});
  elemul_op->LinksFrom({sigmoid_x, sigmoid_out}).LinksTo({elemul_out});
}

}  // namespace patterns

/*
1. fuse sigmoid + elementwise_mul into swish

Origin subgraph:
              input
            /      \
            |      |
            |     sigmoid
            |      |
            |      |
          elementwise_mul
                |
                |
               out

Fused subgraph:
              input
                |
                |
              swish
                |
                |
               out
*/
class SigmoidElementmulXpuFusePass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  const std::string name_scope_{"sigmoid_elementmul_xpu_fuse_pass"};
};

void SigmoidElementmulXpuFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  GraphPatternDetector gpd;
  patterns::SwishXPUPattern pattern(gpd.mutable_pattern(), name_scope_);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle SigmoidElementmulXpuFusePass fuse";

    GET_IR_NODE(sigmoid_x);
    GET_IR_NODE(sigmoid);
    GET_IR_NODE(sigmoid_out);
    GET_IR_NODE(elementwise_mul);
    GET_IR_NODE(elemul_out);

    auto* block = sigmoid->Op()->Block();
    std::string elemul_out_name = elemul_out->Name();

    // Generate swish op
    framework::OpDesc swish_xpu_op_desc(block);
    swish_xpu_op_desc.SetType("swish");
    swish_xpu_op_desc.SetInput("X", {sigmoid_x->Name()});
    swish_xpu_op_desc.SetAttr("beta", 1.f);
    swish_xpu_op_desc.SetOutput("Out", {elemul_out_name});

    auto* swish_xpu = graph->CreateOpNode(&swish_xpu_op_desc);
    IR_NODE_LINK_TO(sigmoid_x, swish_xpu);
    IR_NODE_LINK_TO(swish_xpu, elemul_out);

    // delete useless node
    std::unordered_set<const Node*> delete_nodes;
    delete_nodes = {sigmoid, sigmoid_out, elementwise_mul};
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(sigmoid_elementmul_xpu_fuse_pass,
              paddle::framework::ir::SigmoidElementmulXpuFusePass);

REGISTER_PASS_CAPABILITY(sigmoid_elementmul_xpu_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "swish", 0));
