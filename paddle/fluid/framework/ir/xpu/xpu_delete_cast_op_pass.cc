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

#include "paddle/fluid/framework/ir/xpu/xpu_delete_cast_op_pass.h"
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

int XpuDeleteCastOpPass::ApplyCastSoftmaxPass(ir::Graph* graph) const {
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
                             ->assert_is_op_output("layer_norm", "Y")
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

int XpuDeleteCastOpPass::ApplyCastLayerNormPass(ir::Graph* graph) const {
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

namespace patterns {
struct CastCacheKVInitializationPattern : public PatternBase {
  CastCacheKVInitializationPattern(PDPattern* pattern,
                                   const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(shape0);
  PATTERN_DECL_NODE(shape1);
  PATTERN_DECL_NODE(slice0);
  PATTERN_DECL_NODE(slice1);
  PATTERN_DECL_NODE(cast0);
  PATTERN_DECL_NODE(elementwise_add);
  PATTERN_DECL_NODE(scale);
  PATTERN_DECL_NODE(cast1);
  PATTERN_DECL_NODE(fill_constant);

  // declare variable node's name

  PATTERN_DECL_NODE(shape_in);
  PATTERN_DECL_NODE(shape0_out);
  PATTERN_DECL_NODE(slice0_out);
  PATTERN_DECL_NODE(shape1_out);
  PATTERN_DECL_NODE(slice1_out);
  PATTERN_DECL_NODE(cast0_out);
  PATTERN_DECL_NODE(elementwise_add_in0);
  PATTERN_DECL_NODE(elementwise_add_out);
  PATTERN_DECL_NODE(scale_out);
  PATTERN_DECL_NODE(cast1_out);
};

CastCacheKVInitializationPattern::CastCacheKVInitializationPattern(
    PDPattern* pattern, const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* shape_in =
      pattern->NewNode(shape_in_repr())->assert_is_op_input("shape", "X");

  auto* shape0 = pattern->NewNode(shape0_repr())->assert_is_op("shape");
  auto* shape0_out = pattern->NewNode(shape0_out_repr())
                         ->assert_is_op_output("shape", "Out")
                         ->assert_is_op_input("slice", "X");
  auto* slice0 = pattern->NewNode(slice0_repr())->assert_is_op("slice");
  auto* slice0_out = pattern->NewNode(slice0_out_repr())
                         ->assert_is_op_output("slice", "Out")
                         ->assert_is_op_input("fill_constant", "X");

  auto* shape1 = pattern->NewNode(shape1_repr())->assert_is_op("shape");
  auto* shape1_out = pattern->NewNode(shape1_out_repr())
                         ->assert_is_op_output("shape", "Out")
                         ->assert_is_op_input("slice", "X");
  auto* slice1 = pattern->NewNode(slice1_repr())->assert_is_op("slice");
  auto* slice1_out = pattern->NewNode(slice1_out_repr())
                         ->assert_is_op_output("slice", "Out")
                         ->assert_is_op_input("cast", "X");
  auto* cast0 =
      pattern->NewNode(cast0_repr())
          ->assert_is_op("cast")
          ->assert_more([](Node* node) {
            auto* op_desc = node->Op();
            auto in_dtype = op_desc->GetAttrIfExists<int>("in_dtype");
            auto out_dtype = op_desc->GetAttrIfExists<int>("out_dtype");
            return in_dtype == static_cast<int>(proto::VarType::INT32) &&
                   out_dtype == static_cast<int>(proto::VarType::INT64);
          });
  auto* cast0_out = pattern->NewNode(cast0_out_repr())
                        ->assert_is_op_output("cast", "Out")
                        ->assert_is_op_input("elementwise_add", "Y");
  auto* elementwise_add_in0 = pattern->NewNode(elementwise_add_in0_repr())
                                  ->assert_is_op_input("elementwise_add", "X");
  auto* elementwise_add =
      pattern->NewNode(elementwise_add_repr())->assert_is_op("elementwise_add");
  auto* elementwise_add_out =
      pattern->NewNode(elementwise_add_out_repr())
          ->assert_is_op_output("elementwise_add", "Out")
          ->assert_is_op_input("scale", "X");
  auto* scale = pattern->NewNode(scale_repr())->assert_is_op("scale");

  auto* scale_out = pattern->NewNode(scale_out_repr())
                        ->assert_is_op_output("scale", "Out")
                        ->assert_is_op_input("cast", "X");

  auto* cast1 =
      pattern->NewNode(cast1_repr())
          ->assert_is_op("cast")
          ->assert_more([](Node* node) {
            auto* op_desc = node->Op();
            auto in_dtype = op_desc->GetAttrIfExists<int>("in_dtype");
            auto out_dtype = op_desc->GetAttrIfExists<int>("out_dtype");
            return in_dtype == static_cast<int>(proto::VarType::INT64) &&
                   out_dtype == static_cast<int>(proto::VarType::INT32);
          });
  auto* cast1_out = pattern->NewNode(cast1_out_repr())
                        ->assert_is_op_output("cast", "Out")
                        ->assert_is_op_input("fill_constant", "Y")
                        ->assert_has_n_outputs(1);
  auto* fill_constant =
      pattern->NewNode(fill_constant_repr())->assert_is_op("fill_constant");

  shape0->LinksFrom({shape_in}).LinksTo({shape0_out});
  slice0->LinksFrom({shape0_out}).LinksTo({slice0_out});
  shape1->LinksFrom({shape_in}).LinksTo({shape1_out});
  slice1->LinksFrom({shape1_out}).LinksTo({slice1_out});
  cast0->LinksFrom({slice1_out}).LinksTo({cast0_out});
  elementwise_add->LinksFrom({elementwise_add_in0, cast0_out})
      .LinksTo({elementwise_add_out});
  scale->LinksFrom({elementwise_add_out}).LinksTo({scale_out});
  cast1->LinksFrom({scale_out}).LinksTo({cast1_out});
  fill_constant->LinksFrom({slice0_out, cast1_out});
}
}  // namespace patterns

int XpuDeleteCastOpPass::ApplyCastCacheKVInitializationPass(
    ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::CastCacheKVInitializationPattern pattern(gpd.mutable_pattern(),
                                                     name_scope_);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle ApplyCastCacheKVInitializationPass fuse";
    GET_IR_NODE_FROM_SUBGRAPH(shape_in, shape_in, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(shape0_out, shape0_out, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(slice0_out, slice0_out, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(shape1_out, shape1_out, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(slice1_out, slice1_out, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(cast0_out, cast0_out, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        elementwise_add_in0, elementwise_add_in0, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        elementwise_add_out, elementwise_add_out, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale_out, scale_out, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(cast1_out, cast1_out, pattern);

    GET_IR_NODE_FROM_SUBGRAPH(shape0, shape0, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(slice0, slice0, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(shape1, shape1, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(slice1, slice1, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(cast0, cast0, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(elementwise_add, elementwise_add, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale, scale, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(cast1, cast1, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fill_constant, fill_constant, pattern);

    slice0->Op()->RenameInput(shape0_out->Name(), shape1_out->Name());
    IR_NODE_UNLINK(shape0_out, slice0);
    IR_NODE_LINK_TO(shape1_out, slice0);

    cast1->Op()->RenameInput(scale_out->Name(), elementwise_add_in0->Name());
    elementwise_add->Op()->RenameInput(elementwise_add_in0->Name(),
                                       cast1_out->Name());
    IR_NODE_UNLINK(scale_out, cast1);
    IR_NODE_UNLINK(elementwise_add_in0, elementwise_add);
    IR_NODE_LINK_TO(elementwise_add_in0, cast1);

    fill_constant->Op()->RenameInput(cast1_out->Name(), scale_out->Name());
    IR_NODE_UNLINK(cast1_out, fill_constant);
    IR_NODE_LINK_TO(cast1_out, elementwise_add);
    IR_NODE_LINK_TO(scale_out, fill_constant);

    elementwise_add->Op()->RenameInput(cast0_out->Name(), slice1_out->Name());
    IR_NODE_UNLINK(slice1_out, cast0);
    IR_NODE_UNLINK(cast0_out, elementwise_add);
    IR_NODE_LINK_TO(slice1_out, elementwise_add);

    std::unordered_set<const Node*> delete_nodes{
        shape1, shape0_out, cast0, cast0_out};
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  return found_subgraph_count;
}

void XpuDeleteCastOpPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  if (!graph->IsMainGraph()) {
    VLOG(3) << "'xpu_delete_cast_op_pass' needs info in all "
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

  found_subgraph_count = 0;
  for (size_t i = 0; i < graph->SubGraphsSize(); i++) {
    found_subgraph_count +=
        ApplyCastCacheKVInitializationPass(graph->GetSubGraph(i));
  }
  if (found_subgraph_count > 0) {
    LOG(INFO) << "--- delete " << found_subgraph_count
              << " cast_cachekv_initialization_pattern subgraph";
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(xpu_delete_cast_op_pass,
              paddle::framework::ir::XpuDeleteCastOpPass);

REGISTER_PASS_CAPABILITY(xpu_delete_cast_op_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "cast", 0));
