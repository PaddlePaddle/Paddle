// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/box_coder_multiclass_nms_fuse_pass.h"

namespace paddle {
namespace framework {
namespace ir {

std::unique_ptr<ir::Graph> BoxcoderMultiClassNMSFusePass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  PADDLE_ENFORCE(graph.get());
  FusePassBase::Init("fc_fuse", graph.get());

  std::unordered_set<Node*> nodes2delete;

  GraphPatternDetector gpd;

#define ADD_INPUT(var__, input__, op__)                                     \
  auto* var__ = gpd.mutable_pattern()->NewNode(#var__)->assert_is_op_input( \
      #op__, #input__);

#define ADD_OUTPUT(var__, x__, op__)                                         \
  auto* var__ = gpd.mutable_pattern()->NewNode(#var__)->assert_is_op_output( \
      #op__, #x__);

  ADD_INPUT(target_box, box_coder, TargetBox);
  ADD_INPUT(prior_box, PriorBox, TargetBox);
  ADD_INPUT(prior_box_var, PriorBoxVar, TargetBox);
  ADD_OUTPUT(output_box, box_coder, OutputBox);

  // box corder
  auto* box_coder_op = gpd.mutable_pattern()
                           ->NewNode("box_coder_op")
                           ->assert_is_op("box_corder");

  box_coder_op->LinksFrom({target_box, prior_box, prior_box_var, output_box})
      .LinksTo({output_box});

  // muticlass_nms
  auto* multiclass_nms_op = gpd.mutable_pattern()
                                ->NewNode("multiclass_nms")
                                ->assert_is_op("multiclass_nms");
  ADD_INPUT(bboxes, BBoxes, multi_class_nms);
  ADD_INPUT(scores, Scores, multi_class_nms);
  ADD_INPUT(background_label, background_label, multi_class_nms);
  ADD_INPUT(score_threshold, score_threshold, multi_class_nms);
  ADD_INPUT(nms_top_k, nms_top_k, multi_class_nms);
  ADD_INPUT(nms_threshold, nms_threshold, multi_class_nms);
  ADD_INPUT(nms_eta, nms_eta, multi_class_nms);
  ADD_INPUT(keep_top_k, keep_top_k, multi_class_nms);
  ADD_INPUT(normalized, normalized, multi_class_nms);
  ADD_OUTPUT(nms_out, Out, multi_class_nms);
  multiclass_nms_op
      ->LinksFrom({output_box,  // box_coder's output
                   bboxes, scores, background_label, score_threshold, nms_top_k,
                   nms_threshold, nms_eta, keep_top_k, normalized})
      .LinksTo({nms_out});

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    auto& pattern = gpd.pattern();

#define GET_NODE(n__) auto* n__##_n = subgraph.at(n__);

    // box corder
    GET_NODE(target_box);
    GET_NODE(prior_box);
    GET_NODE(prior_box_var);
    GET_NODE(output_box);

    GET_NODE(box_coder_op);

    // multiclass nms
    GET_NODE(bboxes);
    GET_NODE(scores);
    GET_NODE(background_label);
    GET_NODE(score_threshold);
    GET_NODE(nms_top_k);
    GET_NODE(nms_threshold);
    GET_NODE(nms_eta);
    GET_NODE(keep_top_k);
    GET_NODE(normalized);
    GET_NODE(nms_out);

    GET_NODE(multiclass_nms_op);

    // Construct an operator desc.
    OpDesc desc;
    desc.SetType("detection_out");
    // NOTE fill the real inputs.
    desc.SetInput("input", {target_box_n->Var()->Name()});
    // Create a new Node
    graph->CreateOpNode(&desc);

    // clean graph
    GraphSafeRemoveNodes(graph.get(), {box_coder_op_n, multiclass_nms_op_n});
  };

  gpd(graph.get(), handler);

  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
