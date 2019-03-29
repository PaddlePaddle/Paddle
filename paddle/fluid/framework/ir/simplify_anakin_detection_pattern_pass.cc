// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include <vector>

#include "paddle/fluid/framework/ir/graph_viz_pass.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/ir/simplify_anakin_detection_pattern_pass.h"

namespace paddle {
namespace framework {
namespace ir {

template <int times>
void SimplifyAnakinDetectionPatternPass<times>::ApplyImpl(
    ir::Graph *graph) const {
  const std::string pattern_name =
      "simplify_anakin_detection_pattern_pass" + std::to_string(times);
  FusePassBase::Init(pattern_name, graph);

  GraphPatternDetector gpd;
  std::vector<PDNode *> input_nodes;
  for (int i = 0; i < times; i++) {
    input_nodes.push_back(gpd.mutable_pattern()
                              ->NewNode("x" + std::to_string(i))
                              ->assert_is_op_input("density_prior_box", "Input")
                              ->AsInput());
  }
  input_nodes.push_back(gpd.mutable_pattern()
                            ->NewNode("x" + std::to_string(times))
                            ->assert_is_op_input("box_coder", "TargetBox")
                            ->AsInput());

  input_nodes.push_back(gpd.mutable_pattern()
                            ->NewNode("x" + std::to_string(times + 1))
                            ->assert_is_op_input("transpose2")
                            ->AsInput());

  patterns::AnakinDetectionPattern pattern(gpd.mutable_pattern(), pattern_name);
  pattern(input_nodes, times);

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    const int kNumFields = 7;
    const int kPriorBoxLocOffset = 1;
    const int kReshape1Offset = 2;
    const int kReshape1OutOffset = 3;
    const int kPriorBoxVarOffset = 4;
    const int kReshape2Offset = 5;
    const int kReshape2OutOffset = 6;
    std::vector<Node *> nodes;

    for (int i = 0; i < times; i++) {
      PADDLE_ENFORCE(
          subgraph.at(pattern.GetPDNode("prior_box" + std::to_string(i))));
      PADDLE_ENFORCE(
          subgraph.at(pattern.GetPDNode("box_out" + std::to_string(i))));
      PADDLE_ENFORCE(
          subgraph.at(pattern.GetPDNode("reshape1" + std::to_string(i))));
      PADDLE_ENFORCE(
          subgraph.at(pattern.GetPDNode("reshape1_out" + std::to_string(i))));
      PADDLE_ENFORCE(
          subgraph.at(pattern.GetPDNode("reshape2" + std::to_string(i))));
      PADDLE_ENFORCE(
          subgraph.at(pattern.GetPDNode("reshape2_out" + std::to_string(i))));

      PADDLE_ENFORCE(
          subgraph.at(pattern.GetPDNode("box_var_out" + std::to_string(i))));

      nodes.push_back(
          subgraph.at(pattern.GetPDNode("prior_box" + std::to_string(i))));
      nodes.push_back(
          subgraph.at(pattern.GetPDNode("box_out" + std::to_string(i))));
      nodes.push_back(
          subgraph.at(pattern.GetPDNode("reshape1" + std::to_string(i))));
      nodes.push_back(
          subgraph.at(pattern.GetPDNode("reshape1_out" + std::to_string(i))));
      nodes.push_back(
          subgraph.at(pattern.GetPDNode("box_var_out" + std::to_string(i))));
      nodes.push_back(
          subgraph.at(pattern.GetPDNode("reshape2" + std::to_string(i))));
      nodes.push_back(
          subgraph.at(pattern.GetPDNode("reshape2_out" + std::to_string(i))));
    }

    Node *concat_op1 = subgraph.at(pattern.GetPDNode("concat1"));
    Node *concat_out1 = subgraph.at(pattern.GetPDNode("concat1_out"));

    Node *concat_op2 = subgraph.at(pattern.GetPDNode("concat2"));
    Node *concat_out2 = subgraph.at(pattern.GetPDNode("concat2_out"));

    Node *box_coder_third_input = subgraph.at(input_nodes[times]);
    Node *box_coder_op = subgraph.at(pattern.GetPDNode("box_coder"));
    Node *box_coder_out = subgraph.at(pattern.GetPDNode("box_coder_out"));

    Node *multiclass_nms_second_input = subgraph.at(input_nodes[times + 1]);
    Node *transpose_before_nms =
        subgraph.at(pattern.GetPDNode("transpose_before_nms"));
    Node *transpose_before_nms_out =
        subgraph.at(pattern.GetPDNode("transpose_before_nms_out"));

    Node *multiclass_nms = subgraph.at(pattern.GetPDNode("multiclass_nms"));
    Node *multiclass_nms_out =
        subgraph.at(pattern.GetPDNode("multiclass_nms_out"));

    std::string code_type =
        boost::get<std::string>(box_coder_op->Op()->GetAttr("code_type"));
    bool box_normalized =
        boost::get<bool>(box_coder_op->Op()->GetAttr("box_normalized"));
    // auto variance =
    // boost::get<std::vector<float>>(box_coder_op->Op()->GetAttr("variance"));
    int background_label =
        boost::get<int>(multiclass_nms->Op()->GetAttr("background_label"));
    float score_threshold =
        boost::get<float>(multiclass_nms->Op()->GetAttr("score_threshold"));
    int nms_top_k = boost::get<int>(multiclass_nms->Op()->GetAttr("nms_top_k"));
    float nms_threshold =
        boost::get<float>(multiclass_nms->Op()->GetAttr("nms_threshold"));
    float nms_eta = boost::get<float>(multiclass_nms->Op()->GetAttr("nms_eta"));
    int keep_top_k =
        boost::get<int>(multiclass_nms->Op()->GetAttr("keep_top_k"));

    std::vector<std::string> concat1_input_names;
    for (int i = 0; i < times; i++) {
      concat1_input_names.push_back(
          nodes[i * kNumFields + kPriorBoxLocOffset]->Name());
    }

    // int axis = boost::get<int>(concat_op1->Op()->GetAttr("axis"));
    framework::OpDesc concat1_desc;
    concat1_desc.SetType("concat");
    concat1_desc.SetInput("X", concat1_input_names);
    concat1_desc.SetAttr("axis", 2);
    concat1_desc.SetOutput("Out", {concat_out1->Name()});

    auto *new_add_concat_op = graph->CreateOpNode(&concat1_desc);

    for (int i = 0; i < times; i++) {
      nodes[i * kNumFields + kPriorBoxLocOffset]->outputs.push_back(
          new_add_concat_op);
      new_add_concat_op->inputs.push_back(
          nodes[i * kNumFields + kPriorBoxLocOffset]);
    }

    framework::OpDesc new_op_desc;
    new_op_desc.SetType("detection_out");
    new_op_desc.SetInput("PriorBox", {concat_out1->Name()});
    new_op_desc.SetInput("TargetBox", {box_coder_third_input->Name()});
    new_op_desc.SetInput("Scores", {multiclass_nms_second_input->Name()});
    new_op_desc.SetAttr("code_type", code_type);
    new_op_desc.SetAttr("box_normalized", box_normalized);
    new_op_desc.SetAttr("background_label", background_label);
    new_op_desc.SetAttr("score_threshold", score_threshold);
    new_op_desc.SetAttr("nms_top_k", nms_top_k);
    new_op_desc.SetAttr("nms_threshold", nms_threshold);
    new_op_desc.SetAttr("nms_eta", nms_eta);
    new_op_desc.SetAttr("keep_top_k", keep_top_k);
    new_op_desc.SetOutput("Out", {multiclass_nms_out->Name()});
    new_op_desc.Flush();

    // Create a new node for the fused op.
    auto *detection_out_op = graph->CreateOpNode(&new_op_desc);

    std::unordered_set<const Node *> delete_nodes;

    for (int i = 0; i < times; i++) {
      nodes[i * kNumFields + kPriorBoxLocOffset]->outputs.push_back(concat_op1);
      delete_nodes.insert(nodes[i * kNumFields + kReshape1Offset]);
      delete_nodes.insert(nodes[i * kNumFields + kReshape1OutOffset]);
      delete_nodes.insert(nodes[i * kNumFields + kPriorBoxVarOffset]);
      delete_nodes.insert(nodes[i * kNumFields + kReshape2Offset]);
      delete_nodes.insert(nodes[i * kNumFields + kReshape2OutOffset]);
    }

    delete_nodes.insert(concat_op1);
    delete_nodes.insert(concat_op2);
    delete_nodes.insert(concat_out2);
    delete_nodes.insert(box_coder_op);
    delete_nodes.insert(box_coder_out);
    delete_nodes.insert(transpose_before_nms);
    delete_nodes.insert(transpose_before_nms_out);
    delete_nodes.insert(multiclass_nms);

    new_add_concat_op->outputs.push_back(concat_out1);
    concat_out1->inputs.push_back(new_add_concat_op);

    detection_out_op->inputs.push_back(concat_out1);
    detection_out_op->inputs.push_back(box_coder_third_input);
    detection_out_op->inputs.push_back(multiclass_nms_second_input);
    detection_out_op->outputs.push_back(multiclass_nms_out);

    concat_out1->outputs.push_back(detection_out_op);
    box_coder_third_input->outputs.push_back(detection_out_op);
    multiclass_nms_second_input->outputs.push_back(detection_out_op);
    multiclass_nms_out->inputs.push_back(detection_out_op);

    // Delete the unneeded nodes.
    GraphSafeRemoveNodes(graph, delete_nodes);
  };

  gpd(graph, handler);
}

template class SimplifyAnakinDetectionPatternPass<1>;
template class SimplifyAnakinDetectionPatternPass<2>;
template class SimplifyAnakinDetectionPatternPass<3>;
template class SimplifyAnakinDetectionPatternPass<4>;
template class SimplifyAnakinDetectionPatternPass<5>;
template class SimplifyAnakinDetectionPatternPass<6>;

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(simplify_anakin_detection_pattern_pass,
              paddle::framework::ir::SimplifyAnakinDetectionPatternPass<1>);

REGISTER_PASS(simplify_anakin_detection_pattern_pass2,
              paddle::framework::ir::SimplifyAnakinDetectionPatternPass<2>);

REGISTER_PASS(simplify_anakin_detection_pattern_pass3,
              paddle::framework::ir::SimplifyAnakinDetectionPatternPass<3>);

REGISTER_PASS(simplify_anakin_detection_pattern_pass4,
              paddle::framework::ir::SimplifyAnakinDetectionPatternPass<4>);

REGISTER_PASS(simplify_anakin_detection_pattern_pass5,
              paddle::framework::ir::SimplifyAnakinDetectionPatternPass<5>);

REGISTER_PASS(simplify_anakin_detection_pattern_pass6,
              paddle::framework::ir::SimplifyAnakinDetectionPatternPass<6>);
