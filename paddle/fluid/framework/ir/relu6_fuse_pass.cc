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
#include "paddle/fluid/framework/ir/relu6_fuse_pass.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

struct Relu6Pattern : public PatternBase {
  Relu6Pattern(PDPattern* pattern, const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(clip);
  // declare variable node's name
  PATTERN_DECL_NODE(clip_min);
  PATTERN_DECL_NODE(clip_max);
  PATTERN_DECL_NODE(clip_x);
  PATTERN_DECL_NODE(clip_out);
};

Relu6Pattern::Relu6Pattern(PDPattern* pattern, const std::string& name_scope)
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

Relu6FusePass::Relu6FusePass() {}

void Relu6FusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  GraphPatternDetector gpd;
  patterns::Relu6Pattern pattern(gpd.mutable_pattern(), name_scope_);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle Relu6FusePass fuse";
#define GET_IR_NODE(node_) GET_IR_NODE_FROM_SUBGRAPH(node_, node_, pattern)
    GET_IR_NODE(clip);
    GET_IR_NODE(clip_x);
    GET_IR_NODE(clip_min);
    GET_IR_NODE(clip_max);
    GET_IR_NODE(clip_out);
#undef GET_IR_NODE
    auto* block = clip->Op()->Block();
    auto* scope = param_scope();

    auto clip_min_t =
        scope->Var(clip_min->Name())->GetMutable<phi::DenseTensor>();
    auto clip_max_t =
        scope->Var(clip_max->Name())->GetMutable<phi::DenseTensor>();
    // fp16 --> fp32
    auto tensor_type = clip_min_t->dtype();
    float clip_min_value = 0.f;
    float clip_max_value = 0.f;
    if (tensor_type == phi::DataType::FLOAT16) {
      phi::dtype::float16* clip_min_fp16 =
          clip_min_t->data<phi::dtype::float16>();
      phi::dtype::float16* clip_max_fp16 =
          clip_max_t->data<phi::dtype::float16>();
      clip_min_value = static_cast<float>(clip_min_fp16[0]);
      clip_max_value = static_cast<float>(clip_max_fp16[0]);
    } else if (tensor_type == phi::DataType::FLOAT32) {
      float* clip_min_ptr = clip_min_t->data<float>();
      float* clip_max_ptr = clip_max_t->data<float>();
      clip_min_value = clip_min_ptr[0];
      clip_max_value = clip_max_ptr[0];
    } else {
      VLOG(4) << "The dtype of clip min must be FP32/16, "
                 "but received %d, which is not supported.",
          clip_min_t->dtype();
      return;
    }
    if ((clip_min_value - 0.f) >= 1e-6 || (clip_max_value - 6.f) >= 1e-6)
      return;
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

REGISTER_PASS(relu6_fuse_pass, paddle::framework::ir::Relu6FusePass);

REGISTER_PASS_CAPABILITY(relu6_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "relu6", 0));
