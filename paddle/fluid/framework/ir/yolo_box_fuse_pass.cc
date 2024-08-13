/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/yolo_box_fuse_pass.h"

#include <string>

#include "glog/logging.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle::framework::ir {

class Node;

}  // namespace paddle::framework::ir
namespace paddle::framework::ir::patterns {
struct YoloBoxPattern : public PatternBase {
  YoloBoxPattern(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, name_scope) {
    // elementwise_div pattern
    auto* elt_div_in_x = pattern->NewNode(elt_div_in_x_repr())
                             ->assert_is_op_input("elementwise_div", "X");
    auto* elt_div_in_y = pattern->NewNode(elt_div_in_y_repr())
                             ->assert_is_op_input("elementwise_div", "Y");
    auto* elt_div =
        pattern->NewNode(elt_div_repr())->assert_is_op("elementwise_div");
    auto* elt_div_out = pattern->NewNode(elt_div_out_repr())
                            ->assert_is_op_output("elementwise_div", "Out")
                            ->assert_is_op_input("cast", "X");
    elt_div->LinksFrom({elt_div_in_x, elt_div_in_y}).LinksTo({elt_div_out});
    // cast pattern
    auto* cast = pattern->NewNode(cast_repr())->assert_is_op("cast");
    auto* cast_out = pattern->NewNode(cast_out_repr())
                         ->assert_is_op_output("cast", "Out")
                         ->assert_is_op_input("yolo_box", "ImgSize");
    cast->LinksFrom({elt_div_out}).LinksTo({cast_out});
// 3 * (yolo_box + transpose) pattern
#define YOLO_BOX_TRANSPOSE_PATTERN(idx_)                                       \
  auto* yolo_box##idx_##_in_x = pattern->NewNode(yolo_box##idx_##_in_x_repr()) \
                                    ->assert_is_op_input("yolo_box", "X");     \
  auto* yolo_box##idx_ =                                                       \
      pattern->NewNode(yolo_box##idx_##_repr())->assert_is_op("yolo_box");     \
  auto* yolo_box##idx_##_out_boxes =                                           \
      pattern->NewNode(yolo_box##idx_##_out_boxes_repr())                      \
          ->assert_is_op_output("yolo_box", "Boxes")                           \
          ->assert_is_op_nth_input("concat", "X", idx_);                       \
  auto* yolo_box##idx_##_out_scores =                                          \
      pattern->NewNode(yolo_box##idx_##_out_scores_repr())                     \
          ->assert_is_op_output("yolo_box", "Scores")                          \
          ->assert_is_op_input("transpose2", "X");                             \
  yolo_box##idx_->LinksFrom({yolo_box##idx_##_in_x, cast_out})                 \
      .LinksTo({yolo_box##idx_##_out_boxes, yolo_box##idx_##_out_scores});     \
  auto* transpose##idx_ =                                                      \
      pattern->NewNode(transpose##idx_##_repr())->assert_is_op("transpose2");  \
  auto* transpose##idx_##_out =                                                \
      pattern->NewNode(transpose##idx_##_out_repr())                           \
          ->assert_is_op_output("transpose2", "Out")                           \
          ->assert_is_op_nth_input("concat", "X", idx_);                       \
  auto* transpose##idx_##_out_xshape =                                         \
      pattern->NewNode(transpose##idx_##_out_xshape_repr())                    \
          ->assert_is_op_output("transpose2", "XShape");                       \
  transpose##idx_->LinksFrom({yolo_box##idx_##_out_scores})                    \
      .LinksTo({transpose##idx_##_out, transpose##idx_##_out_xshape});
    YOLO_BOX_TRANSPOSE_PATTERN(0);
    YOLO_BOX_TRANSPOSE_PATTERN(1);
    YOLO_BOX_TRANSPOSE_PATTERN(2);
#undef YOLO_BOX_TRANSPOSE_PATTERN
    // concat0 pattern
    auto* concat0 = pattern->NewNode(concat0_repr())->assert_is_op("concat");
    auto* concat0_out = pattern->NewNode(concat0_out_repr())
                            ->assert_is_op_output("concat", "Out")
                            ->assert_is_op_input("multiclass_nms3", "BBoxes");
    concat0
        ->LinksFrom(
            {yolo_box0_out_boxes, yolo_box1_out_boxes, yolo_box2_out_boxes})
        .LinksTo({concat0_out});
    // concat1 pattern
    auto* concat1 = pattern->NewNode(concat1_repr())->assert_is_op("concat");
    auto* concat1_out = pattern->NewNode(concat1_out_repr())
                            ->assert_is_op_output("concat", "Out")
                            ->assert_is_op_input("multiclass_nms3", "Scores");
    concat1->LinksFrom({transpose0_out, transpose1_out, transpose2_out})
        .LinksTo({concat1_out});
    // nms pattern
    auto* nms = pattern->NewNode(nms_repr())->assert_is_op("multiclass_nms3");
    auto* nms_out = pattern->NewNode(nms_out_repr())
                        ->assert_is_op_output("multiclass_nms3", "Out");
    auto* nms_out_index = pattern->NewNode(nms_out_index_repr())
                              ->assert_is_op_output("multiclass_nms3", "Index");
    auto* nms_out_rois_num =
        pattern->NewNode(nms_out_rois_num_repr())
            ->assert_is_op_output("multiclass_nms3", "NmsRoisNum");
    nms->LinksFrom({concat0_out, concat1_out})
        .LinksTo({nms_out, nms_out_index, nms_out_rois_num});
  }

  // declare operator node's name
  PATTERN_DECL_NODE(elt_div);
  PATTERN_DECL_NODE(cast);
  PATTERN_DECL_NODE(yolo_box0);
  PATTERN_DECL_NODE(yolo_box1);
  PATTERN_DECL_NODE(yolo_box2);
  PATTERN_DECL_NODE(concat0);
  PATTERN_DECL_NODE(transpose0);
  PATTERN_DECL_NODE(transpose1);
  PATTERN_DECL_NODE(transpose2);
  PATTERN_DECL_NODE(concat1);
  PATTERN_DECL_NODE(nms);
  // declare variable node's name
  PATTERN_DECL_NODE(elt_div_in_x);
  PATTERN_DECL_NODE(elt_div_in_y);
  PATTERN_DECL_NODE(elt_div_out);
  PATTERN_DECL_NODE(cast_out);
  PATTERN_DECL_NODE(yolo_box0_in_x);
  PATTERN_DECL_NODE(yolo_box1_in_x);
  PATTERN_DECL_NODE(yolo_box2_in_x);
  PATTERN_DECL_NODE(yolo_box0_out_boxes);
  PATTERN_DECL_NODE(yolo_box1_out_boxes);
  PATTERN_DECL_NODE(yolo_box2_out_boxes);
  PATTERN_DECL_NODE(yolo_box0_out_scores);
  PATTERN_DECL_NODE(yolo_box1_out_scores);
  PATTERN_DECL_NODE(yolo_box2_out_scores);
  PATTERN_DECL_NODE(concat0_out);
  PATTERN_DECL_NODE(transpose0_out);
  PATTERN_DECL_NODE(transpose1_out);
  PATTERN_DECL_NODE(transpose2_out);
  PATTERN_DECL_NODE(transpose0_out_xshape);
  PATTERN_DECL_NODE(transpose1_out_xshape);
  PATTERN_DECL_NODE(transpose2_out_xshape);
  PATTERN_DECL_NODE(concat1_out);
  PATTERN_DECL_NODE(nms_out);
  PATTERN_DECL_NODE(nms_out_index);
  PATTERN_DECL_NODE(nms_out_rois_num);
};
}  // namespace paddle::framework::ir::patterns
namespace paddle::framework::ir {

YoloBoxFusePass::YoloBoxFusePass() = default;

void YoloBoxFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);
  GraphPatternDetector gpd;
  patterns::YoloBoxPattern yolo_box_pattern(gpd.mutable_pattern(), name_scope_);
  int found_subgraph_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle YoloBoxFusePass fuse";
#define GET_IR_NODE(node_) \
  GET_IR_NODE_FROM_SUBGRAPH(node_, node_, yolo_box_pattern)
    GET_IR_NODE(elt_div);
    GET_IR_NODE(cast);
    GET_IR_NODE(yolo_box0);
    GET_IR_NODE(yolo_box1);
    GET_IR_NODE(yolo_box2);
    GET_IR_NODE(concat0);
    GET_IR_NODE(transpose0);
    GET_IR_NODE(transpose1);
    GET_IR_NODE(transpose2);
    GET_IR_NODE(concat1);
    GET_IR_NODE(nms);
    GET_IR_NODE(elt_div_in_x);
    GET_IR_NODE(elt_div_in_y);
    GET_IR_NODE(elt_div_out);
    GET_IR_NODE(cast_out);
    GET_IR_NODE(yolo_box0_in_x);
    GET_IR_NODE(yolo_box1_in_x);
    GET_IR_NODE(yolo_box2_in_x);
    GET_IR_NODE(yolo_box0_out_boxes);
    GET_IR_NODE(yolo_box1_out_boxes);
    GET_IR_NODE(yolo_box2_out_boxes);
    GET_IR_NODE(yolo_box0_out_scores);
    GET_IR_NODE(yolo_box1_out_scores);
    GET_IR_NODE(yolo_box2_out_scores);
    GET_IR_NODE(concat0_out);
    GET_IR_NODE(transpose0_out);
    GET_IR_NODE(transpose1_out);
    GET_IR_NODE(transpose2_out);
    GET_IR_NODE(transpose0_out_xshape);
    GET_IR_NODE(transpose1_out_xshape);
    GET_IR_NODE(transpose2_out_xshape);
    GET_IR_NODE(concat1_out);
    GET_IR_NODE(nms_out);
    GET_IR_NODE(nms_out_index);
    GET_IR_NODE(nms_out_rois_num);
#undef GET_IR_NODE

    auto* block = yolo_box0->Op()->Block();

// create yolo_box_head
#define CREATE_YOLO_BOX_HEAD(idx_)                                         \
  framework::OpDesc yolo_box_head##idx_##_op_desc(block);                  \
  yolo_box_head##idx_##_op_desc.SetType("yolo_box_head");                  \
  yolo_box_head##idx_##_op_desc.SetInput("X",                              \
                                         {yolo_box##idx_##_in_x->Name()}); \
  yolo_box_head##idx_##_op_desc.SetAttr(                                   \
      "anchors", yolo_box##idx_->Op()->GetAttr("anchors"));                \
  yolo_box_head##idx_##_op_desc.SetAttr(                                   \
      "class_num", yolo_box##idx_->Op()->GetAttr("class_num"));            \
  yolo_box_head##idx_##_op_desc.SetOutput(                                 \
      "Out", {yolo_box##idx_##_out_boxes->Name()});                        \
  yolo_box_head##idx_##_op_desc.Flush();                                   \
  auto* yolo_box_head##idx_ =                                              \
      graph->CreateOpNode(&yolo_box_head##idx_##_op_desc);                 \
  IR_NODE_LINK_TO(yolo_box##idx_##_in_x, yolo_box_head##idx_);             \
  IR_NODE_LINK_TO(yolo_box_head##idx_, yolo_box##idx_##_out_boxes);
    CREATE_YOLO_BOX_HEAD(0);
    CREATE_YOLO_BOX_HEAD(1);
    CREATE_YOLO_BOX_HEAD(2);
#undef CREATE_YOLO_BOX_HEAD

    // create yolo_box_post
    framework::OpDesc yolo_box_post_op_desc(block);
    yolo_box_post_op_desc.SetType("yolo_box_post");
    yolo_box_post_op_desc.SetInput("Boxes0", {yolo_box0_out_boxes->Name()});
    yolo_box_post_op_desc.SetInput("Boxes1", {yolo_box1_out_boxes->Name()});
    yolo_box_post_op_desc.SetInput("Boxes2", {yolo_box2_out_boxes->Name()});
    yolo_box_post_op_desc.SetInput("ImageShape", {elt_div_in_x->Name()});
    yolo_box_post_op_desc.SetInput("ImageScale", {elt_div_in_y->Name()});
    yolo_box_post_op_desc.SetAttr("anchors0",
                                  yolo_box0->Op()->GetAttr("anchors"));
    yolo_box_post_op_desc.SetAttr("anchors1",
                                  yolo_box1->Op()->GetAttr("anchors"));
    yolo_box_post_op_desc.SetAttr("anchors2",
                                  yolo_box2->Op()->GetAttr("anchors"));
    yolo_box_post_op_desc.SetAttr("class_num",
                                  yolo_box0->Op()->GetAttr("class_num"));
    yolo_box_post_op_desc.SetAttr("conf_thresh",
                                  yolo_box0->Op()->GetAttr("conf_thresh"));
    yolo_box_post_op_desc.SetAttr("downsample_ratio0",
                                  yolo_box0->Op()->GetAttr("downsample_ratio"));
    yolo_box_post_op_desc.SetAttr("downsample_ratio1",
                                  yolo_box1->Op()->GetAttr("downsample_ratio"));
    yolo_box_post_op_desc.SetAttr("downsample_ratio2",
                                  yolo_box2->Op()->GetAttr("downsample_ratio"));
    yolo_box_post_op_desc.SetAttr("clip_bbox",
                                  yolo_box0->Op()->GetAttr("clip_bbox"));
    yolo_box_post_op_desc.SetAttr("scale_x_y",
                                  yolo_box0->Op()->GetAttr("scale_x_y"));
    yolo_box_post_op_desc.SetAttr("nms_threshold",
                                  nms->Op()->GetAttr("nms_threshold"));
    yolo_box_post_op_desc.SetOutput("Out", {nms_out->Name()});
    yolo_box_post_op_desc.SetOutput("NmsRoisNum", {nms_out_rois_num->Name()});
    auto* yolo_box_post = graph->CreateOpNode(&yolo_box_post_op_desc);
    IR_NODE_LINK_TO(yolo_box0_out_boxes, yolo_box_post);
    IR_NODE_LINK_TO(yolo_box1_out_boxes, yolo_box_post);
    IR_NODE_LINK_TO(yolo_box2_out_boxes, yolo_box_post);
    IR_NODE_LINK_TO(elt_div_in_x, yolo_box_post);
    IR_NODE_LINK_TO(elt_div_in_y, yolo_box_post);
    IR_NODE_LINK_TO(yolo_box_post, nms_out);
    IR_NODE_LINK_TO(yolo_box_post, nms_out_rois_num);

    // delete useless node
    GraphSafeRemoveNodes(graph,
                         {elt_div,
                          cast,
                          yolo_box0,
                          yolo_box1,
                          yolo_box2,
                          concat0,
                          transpose0,
                          transpose1,
                          transpose2,
                          concat1,
                          nms,
                          elt_div_out,
                          cast_out,
                          yolo_box0_out_scores,
                          yolo_box1_out_scores,
                          yolo_box2_out_scores,
                          concat0_out,
                          transpose0_out,
                          transpose1_out,
                          transpose2_out,
                          transpose0_out_xshape,
                          transpose1_out_xshape,
                          transpose2_out_xshape,
                          concat1_out,
                          nms_out_index});
    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

}  // namespace paddle::framework::ir

REGISTER_PASS(yolo_box_fuse_pass, paddle::framework::ir::YoloBoxFusePass);
