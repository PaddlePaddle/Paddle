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

/*
fuse fill_constant + clip block in to relu6 op
For example:
graph:
            Min(0)  Input  Max(6.0)
               \      |     /
                 \    |   /
                    clip
                      |
                      |
                    Output
------------------------------------------------------
After the pass is applied:
                    Input
                      |
                      |
                    relu6
                      |
                      |
                    Output
*/
struct Relu6XPUPattern : public PatternBase {
  Relu6XPUPattern(PDPattern* pattern, const std::string& name_scope);
  // declare operator node's name
  PATTERN_DECL_NODE(clip);
  // declare variable node's name
  PATTERN_DECL_NODE(clip_min);
  PATTERN_DECL_NODE(clip_max);
  PATTERN_DECL_NODE(clip_x);
  PATTERN_DECL_NODE(clip_out);
};

Relu6XPUPattern::Relu6XPUPattern(PDPattern* pattern,
                                 const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto clip = pattern->NewNode(clip_repr())->assert_is_op("clip");

  auto clip_x = pattern->NewNode(clip_x_repr())
                    ->assert_is_op_input("clip", "X")
                    ->assert_var_not_persistable()
                    ->AsInput();
  auto clip_min = pattern->NewNode(clip_min_repr())
                      ->assert_is_op_input("clip", "Min")
                      ->assert_is_persistable_var()
                      ->AsInput();
  auto clip_max = pattern->NewNode(clip_max_repr())
                      ->assert_is_op_input("clip", "Max")
                      ->assert_is_persistable_var()
                      ->AsInput();
  auto clip_out = pattern->NewNode(clip_out_repr())
                      ->assert_is_op_output("clip", "Out")
                      ->assert_has_n_outputs(1);

  clip->LinksFrom({clip_x, clip_min, clip_max}).LinksTo({clip_out});
}

}  // namespace patterns

class Relu6XPUFusePass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  const std::string name_scope_{"relu6_xpu_fuse_pass"};
};

void Relu6XPUFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  GraphPatternDetector gpd;
  patterns::Relu6XPUPattern pattern(gpd.mutable_pattern(), name_scope_);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle Relu6XPUFusePass fuse";
    /* declare operator node's name */
    GET_IR_NODE(clip);
    /* declare variable node's name*/
    GET_IR_NODE(clip_x);
    GET_IR_NODE(clip_min);
    GET_IR_NODE(clip_max);
    GET_IR_NODE(clip_out);
    auto* block = clip->Op()->Block();
    auto* scope = param_scope();
    PADDLE_ENFORCE_NOT_NULL(
        scope, platform::errors::InvalidArgument("Scope cannot be nullptr."));

    auto clip_min_t =
        scope->Var(clip_min->Name())->GetMutable<phi::DenseTensor>();
    auto clip_max_t =
        scope->Var(clip_max->Name())->GetMutable<phi::DenseTensor>();
    float* clip_min_ptr = clip_min_t->data<float>();
    float* clip_max_ptr = clip_max_t->data<float>();
    if (clip_min_ptr[0] != 0.f || clip_max_ptr[0] != 6.f) return;
    // Generate relu6 op
    framework::OpDesc relu6_op_desc(block);
    relu6_op_desc.SetType("relu6");
    // set attrs for fused op
    relu6_op_desc.SetInput("X", {clip_x->Name()});
    relu6_op_desc.SetOutput("Out", {clip_out->Name()});
    // relink fused op
    auto* relu6_op = graph->CreateOpNode(&relu6_op_desc);
    IR_NODE_LINK_TO(clip_x, relu6_op);
    IR_NODE_LINK_TO(relu6_op, clip_out);
    // delete useless node
    std::unordered_set<const Node*> delete_nodes = {clip, clip_min, clip_max};
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(relu6_xpu_fuse_pass, paddle::framework::ir::Relu6XPUFusePass);

REGISTER_PASS_CAPABILITY(relu6_xpu_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "relu6_xpu", 0));
