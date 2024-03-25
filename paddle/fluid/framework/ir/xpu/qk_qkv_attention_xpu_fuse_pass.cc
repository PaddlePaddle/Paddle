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

#include "paddle/fluid/framework/ir/xpu/qk_qkv_attention_xpu_fuse_pass.h"

#include "glog/logging.h"

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/xpu/pass_utils.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

namespace patterns {

struct QkQkvAttentionFusePattern : public PatternBase {
  QkQkvAttentionFusePattern(PDPattern* pattern,
                            const std::string& name_scope,
                            bool with_q_scale);

  // declare operator node's name
  PATTERN_DECL_NODE(reshape_1);
  PATTERN_DECL_NODE(transpose2_1);
  PATTERN_DECL_NODE(slice_1);
  PATTERN_DECL_NODE(slice_2);
  PATTERN_DECL_NODE(slice_3);
  PATTERN_DECL_NODE(scale);
  PATTERN_DECL_NODE(transpose2_2);
  PATTERN_DECL_NODE(qk_matmul);
  PATTERN_DECL_NODE(qk_softmax);
  PATTERN_DECL_NODE(qkv_matmul);
  PATTERN_DECL_NODE(transpose2_3);
  PATTERN_DECL_NODE(reshape_2);

  // declare variable node's name
  PATTERN_DECL_NODE(input);
  PATTERN_DECL_NODE(reshape_1_out);
  PATTERN_DECL_NODE(transpose2_1_out);
  PATTERN_DECL_NODE(slice_1_out);
  PATTERN_DECL_NODE(slice_2_out);
  PATTERN_DECL_NODE(slice_3_out);
  PATTERN_DECL_NODE(scale_out);
  PATTERN_DECL_NODE(transpose2_2_out);
  PATTERN_DECL_NODE(qk_matmul_out);
  PATTERN_DECL_NODE(qk_softmax_out);
  PATTERN_DECL_NODE(qkv_matmul_out);
  PATTERN_DECL_NODE(transpose2_3_out);
  PATTERN_DECL_NODE(output);

 private:
  bool with_q_scale_{false};
};

QkQkvAttentionFusePattern::QkQkvAttentionFusePattern(
    PDPattern* pattern, const std::string& name_scope, bool with_q_scale)
    : PatternBase(pattern, name_scope, name_scope),
      with_q_scale_(with_q_scale) {
  auto* input = pattern->NewNode(input_repr())
                    ->assert_is_op_input("reshape2", "X")
                    ->AsInput();
  auto* reshape_1 =
      pattern->NewNode(reshape_1_repr())->assert_is_op("reshape2");
  auto* reshape_1_out = pattern->NewNode(reshape_1_out_repr())
                            ->assert_is_op_output("reshape2", "Out")
                            ->assert_is_op_input("transpose2", "X");
  auto* transpose2_1 =
      pattern->NewNode(transpose2_1_repr())->assert_is_op("transpose2");
  auto* transpose2_1_out = pattern->NewNode(transpose2_1_out_repr())
                               ->assert_is_op_output("transpose2", "Out")
                               ->assert_is_op_input("slice", "Input");
  auto* slice_1 = pattern->NewNode(slice_1_repr())->assert_is_op("slice");

  PDNode* slice_1_out = nullptr;
  PDNode* scale = nullptr;
  PDNode* scale_out = nullptr;
  if (with_q_scale_) {
    slice_1_out = pattern->NewNode(slice_1_out_repr())
                      ->assert_is_op_output("slice", "Out")
                      ->assert_is_op_input("scale", "X");
    scale = pattern->NewNode(scale_repr())->assert_is_op("scale");
    scale_out = pattern->NewNode(scale_out_repr())
                    ->assert_is_op_output("scale", "Out")
                    ->assert_is_op_input("matmul_v2", "X");
  } else {
    slice_1_out = pattern->NewNode(slice_1_out_repr())
                      ->assert_is_op_output("slice", "Out")
                      ->assert_is_op_input("matmul_v2", "X");
  }
  auto* slice_2 = pattern->NewNode(slice_2_repr())->assert_is_op("slice");
  auto* slice_2_out = pattern->NewNode(slice_2_out_repr())
                          ->assert_is_op_output("slice", "Out")
                          ->assert_is_op_input("transpose2", "X");
  auto* transpose2_2 =
      pattern->NewNode(transpose2_2_repr())
          ->assert_is_op("transpose2")
          ->assert_more([](Node* node) {
            auto* op_desc = node->Op();
            auto axis = op_desc->GetAttrIfExists<std::vector<int>>("axis");
            size_t axis_rank = axis.size();
            return axis_rank == 4 && axis[0] == 0 && axis[1] == 1 &&
                   axis[2] == 3 && axis[3] == 2;
          });
  auto* transpose2_2_out = pattern->NewNode(transpose2_2_out_repr())
                               ->assert_is_op_output("transpose2", "Out")
                               ->assert_is_op_input("matmul_v2", "Y");
  auto* qk_matmul =
      pattern->NewNode(qk_matmul_repr())->assert_is_op("matmul_v2");
  auto* qk_matmul_out = pattern->NewNode(qk_matmul_out_repr())
                            ->assert_is_op_output("matmul_v2", "Out")
                            ->assert_is_op_input("softmax", "X");
  auto* qk_softmax =
      pattern->NewNode(qk_softmax_repr())->assert_is_op("softmax");
  auto* qk_softmax_out = pattern->NewNode(qk_softmax_out_repr())
                             ->assert_is_op_output("softmax", "Out")
                             ->assert_is_op_input("matmul_v2", "X");
  auto* slice_3 = pattern->NewNode(slice_3_repr())->assert_is_op("slice");
  auto* slice_3_out = pattern->NewNode(slice_3_out_repr())
                          ->assert_is_op_output("slice", "Out")
                          ->assert_is_op_input("matmul_v2", "Y");
  auto* qkv_matmul =
      pattern->NewNode(qkv_matmul_repr())->assert_is_op("matmul_v2");
  auto* qkv_matmul_out = pattern->NewNode(qkv_matmul_out_repr())
                             ->assert_is_op_output("matmul_v2", "Out")
                             ->assert_is_op_input("transpose2", "X");
  auto* transpose2_3 =
      pattern->NewNode(transpose2_3_repr())->assert_is_op("transpose2");
  auto* transpose2_3_out = pattern->NewNode(transpose2_3_out_repr())
                               ->assert_is_op_output("transpose2", "Out")
                               ->assert_is_op_input("reshape2", "X");
  auto* reshape_2 =
      pattern->NewNode(reshape_2_repr())->assert_is_op("reshape2");
  auto* output = pattern->NewNode(output_repr())
                     ->AsOutput()
                     ->assert_is_op_output("reshape2", "Out");

  // link nodes
  reshape_1->LinksFrom({input}).LinksTo({reshape_1_out});
  transpose2_1->LinksFrom({reshape_1_out}).LinksTo({transpose2_1_out});
  slice_1->LinksFrom({transpose2_1_out}).LinksTo({slice_1_out});
  slice_2->LinksFrom({transpose2_1_out}).LinksTo({slice_2_out});
  slice_3->LinksFrom({transpose2_1_out}).LinksTo({slice_3_out});
  if (with_q_scale_) {
    scale->LinksFrom({slice_1_out}).LinksTo({scale_out});
    qk_matmul->LinksFrom({scale_out, transpose2_2_out})
        .LinksTo({qk_matmul_out});
  } else {
    qk_matmul->LinksFrom({slice_1_out, transpose2_2_out})
        .LinksTo({qk_matmul_out});
  }
  transpose2_2->LinksFrom({slice_2_out}).LinksTo({transpose2_2_out});
  qk_softmax->LinksFrom({qk_matmul_out}).LinksTo({qk_softmax_out});
  qkv_matmul->LinksFrom({slice_3_out, qk_softmax_out})
      .LinksTo({qkv_matmul_out});
  transpose2_3->LinksFrom({qkv_matmul_out}).LinksTo({transpose2_3_out});
  reshape_2->LinksFrom({transpose2_3_out}).LinksTo({output});
}

}  // namespace patterns

void QkQkvAttentionXPUFusePass::ApplyQkQkvAttentionXPUFuse(
    ir::Graph* graph, bool with_q_scale) const {
  GraphPatternDetector gpd;
  patterns::QkQkvAttentionFusePattern pattern(
      gpd.mutable_pattern(), name_scope_, with_q_scale);
  int found_subgraph_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle QkQkvAttentionXPUFusePass";

    // declare operator node's name
    GET_IR_NODE(reshape_1);
    GET_IR_NODE(transpose2_1);
    GET_IR_NODE(slice_1);
    GET_IR_NODE(slice_2);
    GET_IR_NODE(slice_3);
    GET_IR_NODE(scale);
    GET_IR_NODE(transpose2_2);
    GET_IR_NODE(qk_matmul);
    GET_IR_NODE(qk_softmax);
    GET_IR_NODE(qkv_matmul);
    GET_IR_NODE(transpose2_3);
    GET_IR_NODE(reshape_2);

    // declare variable node's name
    GET_IR_NODE(input);
    GET_IR_NODE(reshape_1_out);
    GET_IR_NODE(transpose2_1_out);
    GET_IR_NODE(slice_1_out);
    GET_IR_NODE(slice_2_out);
    GET_IR_NODE(slice_3_out);
    GET_IR_NODE(scale_out);
    GET_IR_NODE(transpose2_2_out);
    GET_IR_NODE(qk_matmul_out);
    GET_IR_NODE(qk_softmax_out);
    GET_IR_NODE(qkv_matmul_out);
    GET_IR_NODE(transpose2_3_out);
    GET_IR_NODE(output);

    // Generate fuse op
    auto* block = reshape_1->Op()->Block();
    framework::OpDesc fused_op_desc(block);
    fused_op_desc.SetType("qkv_attention_xpu");
    // set input of fuse_op
    fused_op_desc.SetInput("q", {input->Name()});
    fused_op_desc.SetInput("k", {input->Name()});
    fused_op_desc.SetInput("v", {input->Name()});
    // set attributes of fuse_op
    if (with_q_scale) {
      float scale_val = PADDLE_GET_CONST(float, scale->Op()->GetAttr("scale"));
      fused_op_desc.SetAttr("alpha", scale_val);
      VLOG(4) << "while with_q_scale, scale_val = " << scale_val;
    } else {
      // in xdnn, 0.0f is default value of NewBaseAttnParam.alpha
      fused_op_desc.SetAttr("alpha", 0.0f);
    }
    fused_op_desc.SetAttr(
        "head_num", static_cast<int>(transpose2_1_out->Var()->GetShape()[2]));
    fused_op_desc.SetAttr(
        "head_dim", static_cast<int>(transpose2_1_out->Var()->GetShape()[4]));
    // In this pattern, there is only one possible situation.
    fused_op_desc.SetAttr("qkv_fc_fusion", true);

    // TODO(tianrui): support more out_dtype
    fused_op_desc.SetAttr("out_dtype", input->Var()->GetDataType());

    // set output of fuse_op
    VarDesc fused_op_out_max_desc("qkv_max");
    Node* fused_op_out_max = graph->CreateVarNode(&fused_op_out_max_desc);
    fused_op_desc.SetOutput("qkv_max", {"qkv_max"});
    fused_op_desc.SetOutput("qkv", {output->Name()});

    auto* fused_op = graph->CreateOpNode(&fused_op_desc);

    IR_NODE_LINK_TO(input, fused_op);
    IR_NODE_LINK_TO(fused_op, output);
    IR_NODE_LINK_TO(fused_op, fused_op_out_max);

    // delete useless node
    std::unordered_set<const Node*> del_node_set;
    del_node_set.insert(reshape_1);
    del_node_set.insert(reshape_1_out);
    del_node_set.insert(transpose2_1);
    del_node_set.insert(transpose2_1_out);
    del_node_set.insert(slice_1);
    del_node_set.insert(slice_1_out);
    del_node_set.insert(slice_2);
    del_node_set.insert(slice_2_out);
    del_node_set.insert(slice_3);
    del_node_set.insert(slice_3_out);
    del_node_set.insert(scale);
    del_node_set.insert(scale_out);
    del_node_set.insert(transpose2_2);
    del_node_set.insert(transpose2_2_out);
    del_node_set.insert(qk_matmul);
    del_node_set.insert(qk_matmul_out);
    del_node_set.insert(qk_softmax);
    del_node_set.insert(qk_softmax_out);
    del_node_set.insert(qkv_matmul);
    del_node_set.insert(qkv_matmul_out);
    del_node_set.insert(transpose2_3);
    del_node_set.insert(transpose2_3_out);
    del_node_set.insert(reshape_2);
    GraphSafeRemoveNodes(graph, del_node_set);

    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

void QkQkvAttentionXPUFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  for (auto with_q_scale : {true, false}) {
    ApplyQkQkvAttentionXPUFuse(graph, with_q_scale);
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(qk_qkv_attention_xpu_fuse_pass,
              paddle::framework::ir::QkQkvAttentionXPUFusePass);

REGISTER_PASS_CAPABILITY(qk_qkv_attention_xpu_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "qkv_attention_xpu", 0));
