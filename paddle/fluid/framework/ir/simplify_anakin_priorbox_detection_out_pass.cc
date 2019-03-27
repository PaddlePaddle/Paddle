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
#include "paddle/fluid/framework/ir/simplify_anakin_priorbox_detection_out_pass.h"

namespace paddle {
namespace framework {
namespace ir {

template <int times, bool is_density, bool is_reshape>
std::unique_ptr<ir::Graph>
SimplifyAnakinDetectionPatternPass<times, is_density, is_reshape>::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  std::string priorbox_type = is_density ? "density_prior_box" : "prior_box";
  const std::string pattern_name =
      "simplify_anakin_detection_pattern_pass" + std::to_string(times);
  FusePassBase::Init(pattern_name, graph.get());

  GraphPatternDetector gpd;
  std::vector<PDNode *> input_nodes;
  for (int i = 0; i < times; i++) {
    input_nodes.push_back(gpd.mutable_pattern()
                              ->NewNode("x" + std::to_string(i))
                              ->assert_is_op_input(priorbox_type, "Input")
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
  pattern(input_nodes, times, priorbox_type, is_reshape);

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
    GraphSafeRemoveNodes(graph.get(), delete_nodes);
  };

  gpd(graph.get(), handler);
  return graph;
}
// times, priorbox_or_density_priorbox, is_reshape
template class SimplifyAnakinDetectionPatternPass<1, false, true>;
template class SimplifyAnakinDetectionPatternPass<2, false, true>;
template class SimplifyAnakinDetectionPatternPass<3, false, true>;
template class SimplifyAnakinDetectionPatternPass<4, false, true>;
template class SimplifyAnakinDetectionPatternPass<5, false, true>;
template class SimplifyAnakinDetectionPatternPass<6, false, true>;

template class SimplifyAnakinDetectionPatternPass<1, true, true>;
template class SimplifyAnakinDetectionPatternPass<2, true, true>;
template class SimplifyAnakinDetectionPatternPass<3, true, true>;
template class SimplifyAnakinDetectionPatternPass<4, true, true>;
template class SimplifyAnakinDetectionPatternPass<5, true, true>;
template class SimplifyAnakinDetectionPatternPass<6, true, true>;

template class SimplifyAnakinDetectionPatternPass<1, false, false>;
template class SimplifyAnakinDetectionPatternPass<2, false, false>;
template class SimplifyAnakinDetectionPatternPass<3, false, false>;
template class SimplifyAnakinDetectionPatternPass<4, false, false>;
template class SimplifyAnakinDetectionPatternPass<5, false, false>;
template class SimplifyAnakinDetectionPatternPass<6, false, false>;

template class SimplifyAnakinDetectionPatternPass<1, true, false>;
template class SimplifyAnakinDetectionPatternPass<2, true, false>;
template class SimplifyAnakinDetectionPatternPass<3, true, false>;
template class SimplifyAnakinDetectionPatternPass<4, true, false>;
template class SimplifyAnakinDetectionPatternPass<5, true, false>;
template class SimplifyAnakinDetectionPatternPass<6, true, false>;
}  // namespace ir
}  // namespace framework
}  // namespace paddle

typedef paddle::framework::ir::SimplifyAnakinDetectionPatternPass<1, false,
                                                                  true>
    priorbox1_pattern;
REGISTER_PASS(simplify_anakin_priorbox_detection_out_pass, priorbox1_pattern);
typedef paddle::framework::ir::SimplifyAnakinDetectionPatternPass<2, false,
                                                                  true>
    priorbox2_pattern;
REGISTER_PASS(simplify_anakin_priorbox2_detection_out_pass, priorbox2_pattern);
typedef paddle::framework::ir::SimplifyAnakinDetectionPatternPass<3, false,
                                                                  true>
    priorbox3_pattern;
REGISTER_PASS(simplify_anakin_priorbox3_detection_out_pass, priorbox3_pattern);
typedef paddle::framework::ir::SimplifyAnakinDetectionPatternPass<4, false,
                                                                  true>
    priorbox4_pattern;
REGISTER_PASS(simplify_anakin_priorbox4_detection_out_pass, priorbox4_pattern);
typedef paddle::framework::ir::SimplifyAnakinDetectionPatternPass<5, false,
                                                                  true>
    priorbox5_pattern;
REGISTER_PASS(simplify_anakin_priorbox5_detection_out_pass, priorbox5_pattern);
typedef paddle::framework::ir::SimplifyAnakinDetectionPatternPass<6, false,
                                                                  true>
    priorbox6_pattern;
REGISTER_PASS(simplify_anakin_priorbox6_detection_out_pass, priorbox6_pattern);

typedef paddle::framework::ir::SimplifyAnakinDetectionPatternPass<1, true, true>
    densitypriorbox1_pattern;
REGISTER_PASS(simplify_anakin_densitypriorbox1_detection_out_pass,
              densitypriorbox1_pattern);
typedef paddle::framework::ir::SimplifyAnakinDetectionPatternPass<2, true, true>
    densitypriorbox2_pattern;
REGISTER_PASS(simplify_anakin_densitypriorbox2_detection_out_pass,
              densitypriorbox2_pattern);
typedef paddle::framework::ir::SimplifyAnakinDetectionPatternPass<3, true, true>
    densitypriorbox3_pattern;
REGISTER_PASS(simplify_anakin_densitypriorbox3_detection_out_pass,
              densitypriorbox3_pattern);
typedef paddle::framework::ir::SimplifyAnakinDetectionPatternPass<4, true, true>
    densitypriorbox4_pattern;
REGISTER_PASS(simplify_anakin_densitypriorbox4_detection_out_pass,
              densitypriorbox4_pattern);
typedef paddle::framework::ir::SimplifyAnakinDetectionPatternPass<5, true, true>
    densitypriorbox5_pattern;
REGISTER_PASS(simplify_anakin_densitypriorbox5_detection_out_pass,
              densitypriorbox5_pattern);
typedef paddle::framework::ir::SimplifyAnakinDetectionPatternPass<6, true, true>
    densitypriorbox6_pattern;
REGISTER_PASS(simplify_anakin_densitypriorbox6_detection_out_pass,
              densitypriorbox6_pattern);

typedef paddle::framework::ir::SimplifyAnakinDetectionPatternPass<1, false,
                                                                  false>
    priorbox1_flatten_pattern;
REGISTER_PASS(simplify_anakin_priorbox1_flatten_detection_out_pass,
              priorbox1_flatten_pattern);
typedef paddle::framework::ir::SimplifyAnakinDetectionPatternPass<2, false,
                                                                  false>
    priorbox2_flatten_pattern;
REGISTER_PASS(simplify_anakin_priorbox2_flatten_detection_out_pass,
              priorbox2_flatten_pattern);
typedef paddle::framework::ir::SimplifyAnakinDetectionPatternPass<3, false,
                                                                  false>
    priorbox3_flatten_pattern;
REGISTER_PASS(simplify_anakin_priorbox3_flatten_detection_out_pass,
              priorbox3_flatten_pattern);
typedef paddle::framework::ir::SimplifyAnakinDetectionPatternPass<4, false,
                                                                  false>
    priorbox4_flatten_pattern;
REGISTER_PASS(simplify_anakin_priorbox4_flatten_detection_out_pass,
              priorbox4_flatten_pattern);
typedef paddle::framework::ir::SimplifyAnakinDetectionPatternPass<5, false,
                                                                  false>
    priorbox5_flatten_pattern;
REGISTER_PASS(simplify_anakin_priorbox5_flatten_detection_out_pass,
              priorbox5_flatten_pattern);
typedef paddle::framework::ir::SimplifyAnakinDetectionPatternPass<6, false,
                                                                  false>
    priorbox6_flatten_pattern;
REGISTER_PASS(simplify_anakin_priorbox6_flatten_detection_out_pass,
              priorbox6_flatten_pattern);

typedef paddle::framework::ir::SimplifyAnakinDetectionPatternPass<1, true,
                                                                  false>
    densitypriorbox1_flatten_pattern;
REGISTER_PASS(simplify_anakin_densitypriorbox1_flatten_detection_out_pass,
              densitypriorbox1_flatten_pattern);
typedef paddle::framework::ir::SimplifyAnakinDetectionPatternPass<2, true,
                                                                  false>
    densitypriorbox2_flatten_pattern;
REGISTER_PASS(simplify_anakin_densitypriorbox2_flatten_detection_out_pass,
              densitypriorbox2_flatten_pattern);
typedef paddle::framework::ir::SimplifyAnakinDetectionPatternPass<3, true,
                                                                  false>
    densitypriorbox3_flatten_pattern;
REGISTER_PASS(simplify_anakin_densitypriorbox3_flatten_detection_out_pass,
              densitypriorbox3_flatten_pattern);
typedef paddle::framework::ir::SimplifyAnakinDetectionPatternPass<4, true,
                                                                  false>
    densitypriorbox4_flatten_pattern;
REGISTER_PASS(simplify_anakin_densitypriorbox4_flatten_detection_out_pass,
              densitypriorbox4_flatten_pattern);
typedef paddle::framework::ir::SimplifyAnakinDetectionPatternPass<5, true,
                                                                  false>
    densitypriorbox5_flatten_pattern;
REGISTER_PASS(simplify_anakin_densitypriorbox5_flatten_detection_out_pass,
              densitypriorbox5_flatten_pattern);
typedef paddle::framework::ir::SimplifyAnakinDetectionPatternPass<6, true,
                                                                  false>
    densitypriorbox6_flatten_pattern;
REGISTER_PASS(simplify_anakin_densitypriorbox6_flatten_detection_out_pass,
              densitypriorbox6_flatten_pattern);
