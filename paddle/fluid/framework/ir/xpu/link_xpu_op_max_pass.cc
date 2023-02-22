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

struct FusionXPUOpPattern : public PatternBase {
  FusionXPUOpPattern(PDPattern* pattern,
                     const std::string& name_scope,
                     const std::string& op_type);

  // declare operator node's name
  PATTERN_DECL_NODE(fusion_op);
  // declare variable node's name
  PATTERN_DECL_NODE(out);
  PATTERN_DECL_NODE(out_max);

 private:
  std::string op_type_;
};

FusionXPUOpPattern::FusionXPUOpPattern(PDPattern* pattern,
                                       const std::string& name_scope,
                                       const std::string& op_type)
    : PatternBase(pattern, name_scope, name_scope), op_type_(op_type) {
  auto* fusion_op = pattern->NewNode(fusion_op_repr())->assert_is_op(op_type_);
  auto* out = pattern->NewNode(out_repr())
                  ->assert_is_op_output(op_type_, "out")
                  ->assert_var_not_persistable();
  auto* out_max = pattern->NewNode(out_max_repr())
                      ->assert_is_op_output(op_type_, "out_max")
                      ->assert_var_not_persistable();
  fusion_op->LinksTo({out, out_max});
}

}  // namespace patterns

class LinkXPUOpMaxPass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  void ApplyImpl(ir::Graph* graph, const std::string& op_type) const;

  const std::string name_scope_{"multi_encoder_xpu_slice_fuse_pass"};
  // ops with x_max/out_max
  std::set<std::string> op_types_{"fc_xpu", "conv2d_xpu"};
};

/*
Origin subgraph:
          fusion_xpu_op0
            /       \
            |       |
          out0   out0_max
            |
            \
          fusion_xpu_op1

Fused subgraph:
          fusion_xpu_op0
            /       \
            |       |
          out0   out0_max
            |       |
            \       /
          fusion_xpu_op1
*/
void LinkXPUOpMaxPass::ApplyImpl(ir::Graph* graph) const {
  Init(name_scope_, graph);
  for (auto op_type : op_types_) {
    ApplyImpl(graph, op_type);
  }
}

void LinkXPUOpMaxPass::ApplyImpl(ir::Graph* graph,
                                 const std::string& op_type) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  GraphPatternDetector gpd;
  patterns::FusionXPUOpPattern pattern(
      gpd.mutable_pattern(), name_scope_, op_type);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle LinkXPUOpMaxPass fuse";
    GET_IR_NODE(fusion_op);
    GET_IR_NODE(out);
    GET_IR_NODE(out_max);
    for (auto next_op : out->outputs) {
      auto* next_op_desc = next_op->Op();
      if (op_types_.count(next_op_desc->Type()) == 0) continue;
      next_op_desc->SetInput("x_max", {out_max->Name()});
      IR_NODE_LINK_TO(out_max, next_op);
      found_subgraph_count++;
    }
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(link_xpu_op_max_pass, paddle::framework::ir::LinkXPUOpMaxPass);

REGISTER_PASS_CAPABILITY(link_xpu_op_max_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "fc_xpu", 0));
