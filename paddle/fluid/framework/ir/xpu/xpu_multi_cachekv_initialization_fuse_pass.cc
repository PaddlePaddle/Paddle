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

#include "paddle/fluid/framework/ir/xpu/xpu_multi_cachekv_initialization_fuse_pass.h"
#include <string>
#include "glog/logging.h"

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/xpu/pass_utils.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/kernels/cast_kernel.h"

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
struct DeleteMulDataPrePattern : public PatternBase {
  DeleteMulDataPrePattern(PDPattern* pattern, const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(shape_a);
  PATTERN_DECL_NODE(slice0_a);
  PATTERN_DECL_NODE(slice1_a);
  PATTERN_DECL_NODE(cast_a);
  PATTERN_DECL_NODE(elementwise_add_a);
  PATTERN_DECL_NODE(scale_a);
  PATTERN_DECL_NODE(fill_constant_a);

  PATTERN_DECL_NODE(shape_b);
  PATTERN_DECL_NODE(slice0_b);
  PATTERN_DECL_NODE(slice1_b);
  PATTERN_DECL_NODE(cast_b);
  PATTERN_DECL_NODE(elementwise_add_b);
  PATTERN_DECL_NODE(scale_b);
  PATTERN_DECL_NODE(fill_constant_b);
  PATTERN_DECL_NODE(fused_multi_transformer_dyquant_xpu);

  // declare variable node's name
  PATTERN_DECL_NODE(shape_in);
  PATTERN_DECL_NODE(shape_a_out);
  PATTERN_DECL_NODE(slice0_a_out);
  PATTERN_DECL_NODE(slice1_a_out);
  PATTERN_DECL_NODE(cast_a_in);
  PATTERN_DECL_NODE(cast_a_out);
  PATTERN_DECL_NODE(elementwise_add_a_out);
  PATTERN_DECL_NODE(scale_a_out);
  PATTERN_DECL_NODE(fill_constant_a_out);

  PATTERN_DECL_NODE(shape_b_out);
  PATTERN_DECL_NODE(slice0_b_out);
  PATTERN_DECL_NODE(slice1_b_out);
  PATTERN_DECL_NODE(cast_b_in);
  PATTERN_DECL_NODE(cast_b_out);
  PATTERN_DECL_NODE(elementwise_add_b_out);
  PATTERN_DECL_NODE(scale_b_out);
  PATTERN_DECL_NODE(fill_constant_b_out);
};

DeleteMulDataPrePattern::DeleteMulDataPrePattern(PDPattern* pattern,
                                                 const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* shape_in =
      pattern->NewNode(shape_in_repr())->assert_is_op_input("shape", "X");
  // construct sub1
  auto* shape_a = pattern->NewNode(shape_a_repr())->assert_is_op("shape");
  auto* shape_a_out = pattern->NewNode(shape_a_out_repr())
                          ->assert_is_op_output("shape", "Out")
                          ->assert_is_op_input("slice", "X")
                          ->assert_is_op_input("slice", "X");
  auto* slice0_a = pattern->NewNode(slice0_a_repr())->assert_is_op("slice");

  auto* slice0_a_out = pattern->NewNode(slice0_a_out_repr())
                           ->assert_is_op_output("slice", "Out")
                           ->assert_is_op_input("fill_constant", "X");

  auto* slice1_a = pattern->NewNode(slice1_a_repr())->assert_is_op("slice");
  auto* slice1_a_out = pattern->NewNode(slice1_a_out_repr())
                           ->assert_is_op_output("slice", "Out")
                           ->assert_is_op_input("elementwise_add", "Y");
  auto* cast_a_in =
      pattern->NewNode(cast_a_in_repr())->assert_is_op_input("cast", "X");
  auto* cast_a =
      pattern->NewNode(cast_a_repr())
          ->assert_is_op("cast")
          ->assert_more([](Node* node) {
            auto* op_desc = node->Op();
            auto in_dtype = op_desc->GetAttrIfExists<int>("in_dtype");
            auto out_dtype = op_desc->GetAttrIfExists<int>("out_dtype");
            return in_dtype == static_cast<int>(proto::VarType::INT64) &&
                   out_dtype == static_cast<int>(proto::VarType::INT32);
          });
  auto* cast_a_out = pattern->NewNode(cast_a_out_repr())
                         ->assert_is_op_output("cast", "Out")
                         ->assert_is_op_input("elementwise_add", "X")
                         ->assert_has_n_outputs(1);
  auto* elementwise_add_a = pattern->NewNode(elementwise_add_a_repr())
                                ->assert_is_op("elementwise_add");
  auto* elementwise_add_a_out =
      pattern->NewNode(elementwise_add_a_out_repr())
          ->assert_is_op_output("elementwise_add", "Out")
          ->assert_is_op_input("scale", "X");
  auto* scale_a = pattern->NewNode(scale_a_repr())->assert_is_op("scale");

  auto* scale_a_out = pattern->NewNode(scale_a_out_repr())
                          ->assert_is_op_output("scale", "Out")
                          ->assert_is_op_input("fill_constant", "Y");

  auto* fill_constant_a =
      pattern->NewNode(fill_constant_a_repr())->assert_is_op("fill_constant");
  auto* fill_constant_a_out =
      pattern->NewNode(fill_constant_a_out_repr())
          ->assert_is_op_output("fill_constant", "Out")
          ->assert_is_op_input("fused_multi_transformer_dyquant_xpu", "X");
  // construct sub2
  auto* shape_b = pattern->NewNode(shape_b_repr())->assert_is_op("shape");
  auto* shape_b_out = pattern->NewNode(shape_b_out_repr())
                          ->assert_is_op_output("shape", "Out")
                          ->assert_is_op_input("slice", "X")
                          ->assert_is_op_input("slice", "X");
  auto* slice0_b = pattern->NewNode(slice0_b_repr())->assert_is_op("slice");
  auto* slice0_b_out = pattern->NewNode(slice0_b_out_repr())
                           ->assert_is_op_output("slice", "Out")
                           ->assert_is_op_input("fill_constant", "X");

  auto* slice1_b = pattern->NewNode(slice1_b_repr())->assert_is_op("slice");
  auto* slice1_b_out = pattern->NewNode(slice1_b_out_repr())
                           ->assert_is_op_output("slice", "Out")
                           ->assert_is_op_input("elementwise_add", "Y");
  auto* cast_b_in =
      pattern->NewNode(cast_b_in_repr())->assert_is_op_input("cast", "X");
  auto* cast_b =
      pattern->NewNode(cast_b_repr())
          ->assert_is_op("cast")
          ->assert_more([](Node* node) {
            auto* op_desc = node->Op();
            auto in_dtype = op_desc->GetAttrIfExists<int>("in_dtype");
            auto out_dtype = op_desc->GetAttrIfExists<int>("out_dtype");
            return in_dtype == static_cast<int>(proto::VarType::INT64) &&
                   out_dtype == static_cast<int>(proto::VarType::INT32);
          });
  auto* cast_b_out = pattern->NewNode(cast_b_out_repr())
                         ->assert_is_op_output("cast", "Out")
                         ->assert_is_op_input("elementwise_add", "X")
                         ->assert_has_n_outputs(1);
  auto* elementwise_add_b = pattern->NewNode(elementwise_add_b_repr())
                                ->assert_is_op("elementwise_add");
  auto* elementwise_add_b_out =
      pattern->NewNode(elementwise_add_b_out_repr())
          ->assert_is_op_output("elementwise_add", "Out")
          ->assert_is_op_input("scale", "X");
  auto* scale_b = pattern->NewNode(scale_b_repr())->assert_is_op("scale");

  auto* scale_b_out = pattern->NewNode(scale_b_out_repr())
                          ->assert_is_op_output("scale", "Out")
                          ->assert_is_op_input("fill_constant", "Y");

  auto* fill_constant_b =
      pattern->NewNode(fill_constant_b_repr())->assert_is_op("fill_constant");
  auto* fill_constant_b_out =
      pattern->NewNode(fill_constant_b_out_repr())
          ->assert_is_op_output("fill_constant", "Out")
          ->assert_is_op_input("fused_multi_transformer_dyquant_xpu", "Y");

  auto* fused_multi_transformer_dyquant_xpu =
      pattern->NewNode(fused_multi_transformer_dyquant_xpu_repr())
          ->assert_is_op("fused_multi_transformer_dyquant_xpu");

  shape_a->LinksFrom({shape_in}).LinksTo({shape_a_out});
  slice0_a->LinksFrom({shape_a_out}).LinksTo({slice0_a_out});
  slice1_a->LinksFrom({shape_a_out}).LinksTo({slice1_a_out});
  cast_a->LinksFrom({cast_a_in}).LinksTo({cast_a_out});
  elementwise_add_a->LinksFrom({cast_a_out, slice1_a_out})
      .LinksTo({elementwise_add_a_out});
  scale_a->LinksFrom({elementwise_add_a_out}).LinksTo({scale_a_out});
  fill_constant_a->LinksFrom({slice0_a_out, scale_a_out})
      .LinksTo({fill_constant_a_out});

  shape_b->LinksFrom({shape_in}).LinksTo({shape_b_out});
  slice0_b->LinksFrom({shape_b_out}).LinksTo({slice0_b_out});
  slice1_b->LinksFrom({shape_b_out}).LinksTo({slice1_b_out});
  cast_b->LinksFrom({cast_b_in}).LinksTo({cast_b_out});
  elementwise_add_b->LinksFrom({cast_b_out, slice1_b_out})
      .LinksTo({elementwise_add_b_out});
  scale_b->LinksFrom({elementwise_add_b_out}).LinksTo({scale_b_out});
  fill_constant_b->LinksFrom({slice0_b_out, scale_b_out})
      .LinksTo({fill_constant_b_out});

  fused_multi_transformer_dyquant_xpu->LinksFrom(
      {fill_constant_a_out, fill_constant_b_out});
}
}  // namespace patterns

int XpuDeleteMulDataPreparationForFillPass::ApplyDeleteMulDataPrePass(
    ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::DeleteMulDataPrePattern pattern(gpd.mutable_pattern(), name_scope_);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle ApplyDeleteMulDataPrePass fuse";

    GET_IR_NODE_FROM_SUBGRAPH(shape_a, shape_a, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(slice0_a, slice0_a, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(slice1_a, slice1_a, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(cast_a, cast_a, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(elementwise_add_a, elementwise_add_a, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale_a, scale_a, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fill_constant_a, fill_constant_a, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(shape_in, shape_in, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(shape_a_out, shape_a_out, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(slice0_a_out, slice0_a_out, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(slice1_a_out, slice1_a_out, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(cast_a_in, cast_a_in, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(cast_a_out, cast_a_out, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        elementwise_add_a_out, elementwise_add_a_out, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale_a_out, scale_a_out, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        fill_constant_a_out, fill_constant_a_out, pattern);

    GET_IR_NODE_FROM_SUBGRAPH(shape_b, shape_b, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(slice0_b, slice0_b, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(slice1_b, slice1_b, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(cast_b, cast_b, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(elementwise_add_b, elementwise_add_b, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale_b, scale_b, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fill_constant_b, fill_constant_b, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(shape_b_out, shape_b_out, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(slice0_b_out, slice0_b_out, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(slice1_b_out, slice1_b_out, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(cast_b_in, cast_b_in, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(cast_b_out, cast_b_out, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        elementwise_add_b_out, elementwise_add_b_out, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale_b_out, scale_b_out, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        fill_constant_b_out, fill_constant_b_out, pattern);

    GET_IR_NODE_FROM_SUBGRAPH(fused_multi_transformer_dyquant_xpu,
                              fused_multi_transformer_dyquant_xpu,
                              pattern);

    if (slice0_a->Op()->GetAttrIfExists<std::vector<int>>("axes") ==
            slice0_b->Op()->GetAttrIfExists<std::vector<int>>("axes") &&
        slice0_a->Op()->GetAttrIfExists<std::vector<int>>("starts") ==
            slice0_b->Op()->GetAttrIfExists<std::vector<int>>("starts") &&
        slice0_a->Op()->GetAttrIfExists<std::vector<int>>("ends") ==
            slice0_b->Op()->GetAttrIfExists<std::vector<int>>("ends") &&
        slice1_a->Op()->GetAttrIfExists<std::vector<int>>("axes") ==
            slice1_b->Op()->GetAttrIfExists<std::vector<int>>("axes") &&
        slice1_a->Op()->GetAttrIfExists<std::vector<int>>("starts") ==
            slice1_b->Op()->GetAttrIfExists<std::vector<int>>("starts") &&
        slice1_a->Op()->GetAttrIfExists<std::vector<int>>("ends") ==
            slice1_b->Op()->GetAttrIfExists<std::vector<int>>("ends")) {
      fill_constant_b->Op()->RenameInput(slice0_b_out->Name(),
                                         slice0_a_out->Name());
      fill_constant_b->Op()->RenameInput(scale_b_out->Name(),
                                         scale_a_out->Name());
      IR_NODE_UNLINK(slice0_b_out, fill_constant_b);
      IR_NODE_UNLINK(scale_b_out, fill_constant_b);
      IR_NODE_LINK_TO(slice0_a_out, fill_constant_b);
      IR_NODE_LINK_TO(scale_a_out, fill_constant_b);

      std::unordered_set<const Node*> delete_nodes{shape_b,
                                                   slice0_b,
                                                   slice1_b,
                                                   cast_b,
                                                   elementwise_add_b,
                                                   scale_b,
                                                   shape_b_out,
                                                   slice0_b_out,
                                                   slice1_b_out,
                                                   cast_b_in,
                                                   cast_b_out,
                                                   elementwise_add_b_out,
                                                   scale_b_out};
      GraphSafeRemoveNodes(graph, delete_nodes);
      found_subgraph_count++;
    }
  };

  gpd(graph, handler);
  return found_subgraph_count;
}

void XpuDeleteMulDataPreparationForFillPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  if (!graph->IsMainGraph()) {
    VLOG(3) << "'xpu_multi_cachekv_initialization_fuse_pass' needs info in all "
               "graphs, so it "
               "should be applied in the main graph.";
    return;
  }
  Init(name_scope_, graph);

  int found_subgraph_count = ApplyDeleteMulDataPrePass(graph);
  for (size_t i = 0; i < graph->SubGraphsSize(); i++) {
    found_subgraph_count += ApplyDeleteMulDataPrePass(graph->GetSubGraph(i));
  }
  if (found_subgraph_count > 0) {
    LOG(INFO) << "--- delete " << found_subgraph_count
              << " optimize_data subgraph";
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(xpu_multi_cachekv_initialization_fuse_pass,
              paddle::framework::ir::XpuDeleteMulDataPreparationForFillPass);

REGISTER_PASS_CAPABILITY(xpu_multi_cachekv_initialization_fuse_pass);
