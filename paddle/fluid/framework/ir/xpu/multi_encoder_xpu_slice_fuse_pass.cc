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

struct MultiEncoderXPUSlicePattern : public PatternBase {
  MultiEncoderXPUSlicePattern(PDPattern* pattern,
                              const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(multi_encoder_xpu);
  PATTERN_DECL_NODE(slice);
  // declare variable node's name
  PATTERN_DECL_NODE(multi_encoder_xpu_out);
  PATTERN_DECL_NODE(slice_out);
};

MultiEncoderXPUSlicePattern::MultiEncoderXPUSlicePattern(
    PDPattern* pattern, const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* multi_encoder_xpu =
      pattern->NewNode(multi_encoder_xpu_repr())
          ->assert_is_op("multi_encoder_xpu")
          ->assert_more([](Node* node) {
            return (!PADDLE_GET_CONST(bool,
                                      node->Op()->GetAttr("norm_before"))) &&
                   (PADDLE_GET_CONST(int, node->Op()->GetAttr("slice_idx")) ==
                    -1);
          });
  auto* multi_encoder_xpu_out =
      pattern->NewNode(multi_encoder_xpu_out_repr())
          ->assert_is_op_output("multi_encoder_xpu", "out")
          ->assert_is_op_input("slice", "Input")
          ->assert_var_not_persistable()
          ->assert_has_n_outputs(1);
  auto* slice =
      pattern->NewNode(slice_repr())
          ->assert_is_op("slice")
          ->assert_more([](Node* node) {
            std::vector<int> axes =
                PADDLE_GET_CONST(std::vector<int>, node->Op()->GetAttr("axes"));
            std::vector<int> starts = PADDLE_GET_CONST(
                std::vector<int>, node->Op()->GetAttr("starts"));
            std::vector<int> ends =
                PADDLE_GET_CONST(std::vector<int>, node->Op()->GetAttr("ends"));
            return axes.size() == 1 && axes[0] == 1 && starts.size() == 1 &&
                   starts[0] == 0 &&  //
                   ends.size() == 1 && ends[0] == 1;
          });
  auto* slice_out = pattern->NewNode(slice_out_repr())
                        ->assert_is_op_output("slice", "Out")
                        ->assert_var_not_persistable();
  multi_encoder_xpu->LinksTo({multi_encoder_xpu_out});
  slice->LinksFrom({multi_encoder_xpu_out}).LinksTo({slice_out});
}

}  // namespace patterns

/*
Origin subgraph:
          multi_encoder_xpu
                  |
                slice

Fused subgraph:
          multi_encoder_xpu
*/
class MultiEncoderXPUSliceFusePass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  const std::string name_scope_{"multi_encoder_xpu_slice_fuse_pass"};
};

void MultiEncoderXPUSliceFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);
  GraphPatternDetector gpd;
  patterns::MultiEncoderXPUSlicePattern pattern(gpd.mutable_pattern(),
                                                name_scope_);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle MultiEncoderXPUSliceFusePass fuse";
    GET_IR_NODE(multi_encoder_xpu);
    GET_IR_NODE(slice);
    GET_IR_NODE(multi_encoder_xpu_out);
    GET_IR_NODE(slice_out);

    auto* op_desc = multi_encoder_xpu->Op();
    op_desc->SetOutput("out", {slice_out->Var()->Name()});
    op_desc->SetAttr("slice_idx", static_cast<int>(0));
    IR_NODE_LINK_TO(multi_encoder_xpu, slice_out);

    // delete useless node
    std::unordered_set<const Node*> delete_nodes{multi_encoder_xpu_out, slice};
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(multi_encoder_xpu_slice_fuse_pass,
              paddle::framework::ir::MultiEncoderXPUSliceFusePass);

REGISTER_PASS_CAPABILITY(multi_encoder_xpu_slice_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "multi_encoder_xpu", 0));
