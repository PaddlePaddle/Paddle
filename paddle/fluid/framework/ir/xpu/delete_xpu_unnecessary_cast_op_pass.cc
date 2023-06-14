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

#include "paddle/fluid/framework/ir/xpu/delete_xpu_unnecessary_cast_op_pass.h"
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
struct CastSoftmaxPattern : public PatternBase {
  CastSoftmaxPattern(PDPattern* pattern, const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(cast0);
  PATTERN_DECL_NODE(softmax);
  PATTERN_DECL_NODE(cast1);
  // declare variable node's name
  PATTERN_DECL_NODE(cast0_in);
  PATTERN_DECL_NODE(cast0_out);
  PATTERN_DECL_NODE(softmax_out);
  PATTERN_DECL_NODE(cast1_out);
};

CastSoftmaxPattern::CastSoftmaxPattern(PDPattern* pattern,
                                       const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* cast0_in =
      pattern->NewNode(cast0_in_repr())->assert_is_op_input("cast", "X");
  auto* cast0 =
      pattern->NewNode(cast0_repr())
          ->assert_is_op("cast")
          ->assert_more([](Node* node) {
            auto* op_desc = node->Op();
            auto in_dtype = op_desc->GetAttrIfExists<int>("in_dtype");
            auto out_dtype = op_desc->GetAttrIfExists<int>("out_dtype");
            return in_dtype == static_cast<int>(proto::VarType::FP16) &&
                   out_dtype == static_cast<int>(proto::VarType::FP32);
          });
  auto* cast0_out = pattern->NewNode(cast0_out_repr())
                        ->assert_is_op_output("cast", "Out")
                        ->assert_is_op_input("softmax", "X")
                        ->assert_has_n_outputs(1);
  auto* softmax = pattern->NewNode(softmax_repr())->assert_is_op("softmax");
  auto* softmax_out = pattern->NewNode(softmax_out_repr())
                          ->assert_is_op_output("softmax", "Out")
                          ->assert_is_op_input("cast", "X")
                          ->assert_has_n_outputs(1);
  auto* cast1 =
      pattern->NewNode(cast1_repr())
          ->assert_is_op("cast")
          ->assert_more([](Node* node) {
            auto* op_desc = node->Op();
            auto in_dtype = op_desc->GetAttrIfExists<int>("in_dtype");
            auto out_dtype = op_desc->GetAttrIfExists<int>("out_dtype");
            return in_dtype == static_cast<int>(proto::VarType::FP32) &&
                   out_dtype == static_cast<int>(proto::VarType::FP16);
          });
  auto* cast1_out =
      pattern->NewNode(cast1_out_repr())->assert_is_op_output("cast", "Out");

  cast0->LinksFrom({cast0_in}).LinksTo({cast0_out});
  softmax->LinksFrom({cast0_out}).LinksTo({softmax_out});
  cast1->LinksFrom({softmax_out}).LinksTo({cast1_out});
}
}  // namespace patterns

int DeleteXpuUnnecessaryCastOpPass::ApplyCastSoftmaxPass(
    ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::CastSoftmaxPattern pattern(gpd.mutable_pattern(), name_scope_);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle ApplyCastSoftmaxPass fuse";
    GET_IR_NODE_FROM_SUBGRAPH(cast0, cast0, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(softmax, softmax, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(cast1, cast1, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(cast0_in, cast0_in, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(cast0_out, cast0_out, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(softmax_out, softmax_out, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(cast1_out, cast1_out, pattern);

    softmax->Op()->RenameInput(cast0_out->Name(), cast0_in->Name());
    softmax->Op()->RenameOutput(softmax_out->Name(), cast1_out->Name());
    IR_NODE_LINK_TO(cast0_in, softmax);
    IR_NODE_LINK_TO(softmax, cast1_out);

    std::unordered_set<const Node*> delete_nodes{
        cast0, cast1, cast0_out, softmax_out};
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  return found_subgraph_count;
}

namespace patterns {
struct CastLayerNormPattern : public PatternBase {
  CastLayerNormPattern(PDPattern* pattern, const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(cast0);
  PATTERN_DECL_NODE(layer_norm);
  PATTERN_DECL_NODE(cast1);
  // declare variable node's name
  PATTERN_DECL_NODE(cast0_in);
  PATTERN_DECL_NODE(cast0_out);
  PATTERN_DECL_NODE(layer_norm_out);
  PATTERN_DECL_NODE(cast1_out);
};

CastLayerNormPattern::CastLayerNormPattern(PDPattern* pattern,
                                           const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* cast0_in =
      pattern->NewNode(cast0_in_repr())->assert_is_op_input("cast", "X");
  auto* cast0 =
      pattern->NewNode(cast0_repr())
          ->assert_is_op("cast")
          ->assert_more([](Node* node) {
            auto* op_desc = node->Op();
            auto in_dtype = op_desc->GetAttrIfExists<int>("in_dtype");
            auto out_dtype = op_desc->GetAttrIfExists<int>("out_dtype");
            return in_dtype == static_cast<int>(proto::VarType::FP16) &&
                   out_dtype == static_cast<int>(proto::VarType::FP32);
          });
  auto* cast0_out = pattern->NewNode(cast0_out_repr())
                        ->assert_is_op_output("cast", "Out")
                        ->assert_is_op_input("layer_norm", "X")
                        ->assert_has_n_outputs(1);
  auto* layer_norm =
      pattern->NewNode(layer_norm_repr())->assert_is_op("layer_norm");
  auto* layer_norm_out = pattern->NewNode(layer_norm_out_repr())
                             ->assert_is_op_output("layer_norm", "Out")
                             ->assert_is_op_input("cast", "X")
                             ->assert_has_n_outputs(1);
  auto* cast1 =
      pattern->NewNode(cast1_repr())
          ->assert_is_op("cast")
          ->assert_more([](Node* node) {
            auto* op_desc = node->Op();
            auto in_dtype = op_desc->GetAttrIfExists<int>("in_dtype");
            auto out_dtype = op_desc->GetAttrIfExists<int>("out_dtype");
            return in_dtype == static_cast<int>(proto::VarType::FP32) &&
                   out_dtype == static_cast<int>(proto::VarType::FP16);
          });
  auto* cast1_out =
      pattern->NewNode(cast1_out_repr())->assert_is_op_output("cast", "Out");

  cast0->LinksFrom({cast0_in}).LinksTo({cast0_out});
  layer_norm->LinksFrom({cast0_out}).LinksTo({layer_norm_out});
  cast1->LinksFrom({layer_norm_out}).LinksTo({cast1_out});
}
}  // namespace patterns

int DeleteXpuUnnecessaryCastOpPass::ApplyCastLayerNormPass(
    ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::CastLayerNormPattern pattern(gpd.mutable_pattern(), name_scope_);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle ApplyCastLayerNormPass fuse";
    GET_IR_NODE_FROM_SUBGRAPH(cast0, cast0, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm, layer_norm, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(cast1, cast1, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(cast0_in, cast0_in, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(cast0_out, cast0_out, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_out, layer_norm_out, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(cast1_out, cast1_out, pattern);

    layer_norm->Op()->RenameInput(cast0_out->Name(), cast0_in->Name());
    layer_norm->Op()->RenameOutput(layer_norm_out->Name(), cast1_out->Name());
    IR_NODE_LINK_TO(cast0_in, layer_norm);
    IR_NODE_LINK_TO(layer_norm, cast1_out);

    std::unordered_set<const Node*> delete_nodes{
        cast0, cast1, cast0_out, layer_norm_out};
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  return found_subgraph_count;
}

void DeleteXpuUnnecessaryCastOpPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  if (!graph->IsMainGraph()) {
    VLOG(3) << "'delete_xpu_unnecessary_cast_op_pass' needs info in all "
               "graphs, so it "
               "should be applied in the main graph.";
    return;
  }
  Init(name_scope_, graph);

  int found_subgraph_count = ApplyCastSoftmaxPass(graph);
  for (size_t i = 0; i < graph->SubGraphsSize(); i++) {
    found_subgraph_count += ApplyCastSoftmaxPass(graph->GetSubGraph(i));
  }
  if (found_subgraph_count > 0) {
    LOG(INFO) << "--- delete " << found_subgraph_count
              << " cast_softmax_cast subgraph";
  }

  found_subgraph_count = 0;
  for (size_t i = 0; i < graph->SubGraphsSize(); i++) {
    found_subgraph_count += ApplyCastLayerNormPass(graph->GetSubGraph(i));
  }
  if (found_subgraph_count > 0) {
    LOG(INFO) << "--- delete " << found_subgraph_count
              << " cast_layer_norm_cast subgraph";
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(delete_xpu_unnecessary_cast_op_pass,
              paddle::framework::ir::DeleteXpuUnnecessaryCastOpPass);

REGISTER_PASS_CAPABILITY(delete_xpu_unnecessary_cast_op_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "cast", 0));
