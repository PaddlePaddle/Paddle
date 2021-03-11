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

#include "paddle/fluid/framework/ir/embedding_eltwise_layernorm_fuse_pass.h"

#include <string>

#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
namespace ir {
class Node;
}  // namespace ir
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

static PDNode* create_emb_vars(PDPattern* pattern, const std::string& name,
                               const std::string& arg,
                               bool is_persist = false) {
  PDNode* node =
      pattern->NewNode(name)->assert_is_op_input("lookup_table", arg);
  if (is_persist) return node->assert_is_persistable_var();
  return node;
}
static PDNode* create_emb_out_vars(PDPattern* pattern, const std::string& name,
                                   const std::string& arg) {
  PDNode* node = pattern->NewNode(name)
                     ->assert_is_only_output_of_op("lookup_table")
                     ->assert_is_op_input("elementwise_add", arg)
                     ->AsIntermediate();
  return node;
}
void Embedding2Eltwise1Pattern::operator()() {
  auto* lookup_table1_x =
      create_emb_vars(pattern, lookup_table1_x_repr(), "Ids");
  auto* lookup_table2_x =
      create_emb_vars(pattern, lookup_table2_x_repr(), "Ids");
  auto* lookup_table1_w =
      create_emb_vars(pattern, lookup_table1_w_repr(), "W", true);
  auto* lookup_table2_w =
      create_emb_vars(pattern, lookup_table2_w_repr(), "W", true);
  auto* lookup_table1 =
      pattern->NewNode(lookup_table1_repr())->assert_is_op("lookup_table");
  auto* lookup_table2 =
      pattern->NewNode(lookup_table2_repr())->assert_is_op("lookup_table");
  auto* lookup_table1_out =
      create_emb_out_vars(pattern, lookup_table1_out_repr(), "X");
  auto* lookup_table2_out =
      create_emb_out_vars(pattern, lookup_table2_out_repr(), "Y");
  auto* eltwise_add =
      pattern->NewNode(eltwise_add_repr())->assert_is_op("elementwise_add");
  auto* eltwise_add_out = pattern->NewNode(eltwise_add_out_repr())
                              ->assert_is_op_output("elementwise_add");
  lookup_table1->LinksFrom({lookup_table1_x, lookup_table1_w})
      .LinksTo({lookup_table1_out});
  lookup_table2->LinksFrom({lookup_table2_x, lookup_table2_w})
      .LinksTo({lookup_table2_out});
  eltwise_add->LinksFrom({lookup_table1_out, lookup_table2_out})
      .LinksTo({eltwise_add_out});
}
void Embedding1Eltwise1Pattern::operator()() {
  auto* lookup_table1_x =
      create_emb_vars(pattern, lookup_table1_x_repr(), "Ids");
  auto* lookup_table1_w =
      create_emb_vars(pattern, lookup_table1_w_repr(), "W", true);
  auto* lookup_table1 =
      pattern->NewNode(lookup_table1_repr())->assert_is_op("lookup_table");
  auto* lookup_table1_out =
      create_emb_out_vars(pattern, lookup_table1_out_repr(), "Y");
  auto* eltwise_add =
      pattern->NewNode(eltwise_add_repr())->assert_is_op("elementwise_add");
  auto* eltwise_add_in = pattern->NewNode(eltwise_add_in_repr())
                             ->assert_is_op_input("elementwise_add", "X")
                             ->assert_is_op_output("elementwise_add");
  auto* eltwise_add_out = pattern->NewNode(eltwise_add_out_repr())
                              ->assert_is_op_output("elementwise_add");
  lookup_table1->LinksFrom({lookup_table1_x, lookup_table1_w})
      .LinksTo({lookup_table1_out});
  eltwise_add->LinksFrom({lookup_table1_out, eltwise_add_in})
      .LinksTo({eltwise_add_out});
}
void SkipLayerNorm::operator()() {
  auto* eltwise_add =
      pattern->NewNode(eltwise_add_repr())->assert_is_op("elementwise_add");
  auto* eltwise_add_out = pattern->NewNode(eltwise_add_out_repr())
                              ->assert_is_op_output("elementwise_add")
                              ->assert_is_op_input("layer_norm", "X")
                              ->AsIntermediate();
  auto* layer_norm =
      pattern->NewNode(layer_norm_repr())->assert_is_op("layer_norm");
  auto* layer_norm_out = pattern->NewNode(layer_norm_out_repr())
                             ->assert_is_op_output("layer_norm", "Y")
                             ->AsOutput();
  auto* layer_norm_bias_var = pattern->NewNode(layer_norm_bias_repr())
                                  ->AsInput()
                                  ->assert_is_persistable_var()
                                  ->assert_is_op_input("layer_norm", "Bias");
  auto* layer_norm_scale_var = pattern->NewNode(layer_norm_scale_repr())
                                   ->AsInput()
                                   ->assert_is_persistable_var()
                                   ->assert_is_op_input("layer_norm", "Scale");
  auto* layer_norm_mean_var = pattern->NewNode(layer_norm_mean_repr())
                                  ->AsOutput()
                                  ->assert_is_op_output("layer_norm", "Mean");
  auto* layer_norm_variance_var =
      pattern->NewNode(layer_norm_variance_repr())
          ->AsOutput()
          ->assert_is_op_output("layer_norm", "Variance");
  eltwise_add->LinksTo({eltwise_add_out});
  layer_norm
      ->LinksFrom({eltwise_add_out, layer_norm_bias_var, layer_norm_scale_var})
      .LinksTo({layer_norm_out, layer_norm_mean_var, layer_norm_variance_var});
}
static int BuildFusion(Graph* graph, const std::string& name_scope
                       /*const Scope* scope*/) {
  GraphPatternDetector gpd;
  auto* pattern = gpd.mutable_pattern();

  std::vector<std::vector<std::pair<Node*, Node*>>> start_pattern_in_nodes;
  std::vector<Node*> start_pattern_out_node;
  std::vector<std::unordered_set<Node*>> start_pattern_remove_nodes;

  // Create pattern.
  Embedding2Eltwise1Pattern start_pattern(pattern, name_scope + "/start");
  start_pattern();
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(lookup_table1_x, lookup_table1_x, start_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(lookup_table2_x, lookup_table2_x, start_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(lookup_table1_w, lookup_table1_w, start_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(lookup_table2_w, lookup_table2_w, start_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(lookup_table1, lookup_table1, start_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(lookup_table2, lookup_table2, start_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(lookup_table1_out, lookup_table1_out,
                              start_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(lookup_table2_out, lookup_table2_out,
                              start_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltwise_add, eltwise_add, start_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltwise_add_out, eltwise_add_out, start_pattern);
    std::vector<std::pair<Node*, Node*>> ins;
    ins.push_back(std::make_pair(lookup_table1_x, lookup_table1_w));
    ins.push_back(std::make_pair(lookup_table2_x, lookup_table2_w));
    start_pattern_in_nodes.push_back(ins);
    start_pattern_out_node.push_back(eltwise_add_out);

    std::unordered_set<Node*> rm_nodes;
    rm_nodes.insert({lookup_table1, lookup_table2, lookup_table1_out,
                     lookup_table2_out, eltwise_add, eltwise_add_out});
    start_pattern_remove_nodes.push_back(rm_nodes);
  };
  gpd(graph, handler);

  std::vector<std::pair<Node*, Node*>> inner_pattern_ins;
  std::vector<Node*> inner_pattern_tmp_in;
  std::vector<Node*> inner_pattern_out;
  std::vector<std::unordered_set<Node*>> inner_pattern_remove_nodes;

  GraphPatternDetector gpd2;
  auto* pattern2 = gpd2.mutable_pattern();
  Embedding1Eltwise1Pattern second_pattern(pattern2, name_scope + "/second");
  second_pattern();
  auto handler2 = [&](const GraphPatternDetector::subgraph_t& subgraph,
                      Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(lookup_table1_x, lookup_table1_x, second_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(lookup_table1_w, lookup_table1_w, second_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(lookup_table1, lookup_table1, second_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(lookup_table1_out, lookup_table1_out,
                              second_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltwise_add_in, eltwise_add_in, second_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltwise_add, eltwise_add, second_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltwise_add_out, eltwise_add_out, second_pattern);
    auto in = std::make_pair(lookup_table1_x, lookup_table1_w);
    inner_pattern_ins.push_back(in);
    inner_pattern_tmp_in.push_back(eltwise_add_in);
    inner_pattern_out.push_back(eltwise_add_out);

    std::unordered_set<Node*> rm_nodes;
    rm_nodes.insert(
        {lookup_table1, lookup_table1_out, eltwise_add, eltwise_add_out});
    inner_pattern_remove_nodes.push_back(rm_nodes);
  };
  gpd2(graph, handler2);

  std::vector<Node*> end_pattern_elt_out;
  std::vector<Node*> end_pattern_scales;
  std::vector<Node*> end_pattern_biases;
  std::vector<Node*> end_pattern_out;
  std::vector<Node*> end_patter_layernorms;
  std::vector<std::unordered_set<Node*>> end_pattern_remove_nodes;
  GraphPatternDetector gpd3;
  auto* pattern3 = gpd3.mutable_pattern();
  SkipLayerNorm skip_layernorm_pattern(pattern3, name_scope + "/third");
  skip_layernorm_pattern();
  auto handler3 = [&](const GraphPatternDetector::subgraph_t& subgraph,
                      Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(eltwise_add, eltwise_add, skip_layernorm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltwise_add_out, eltwise_add_out,
                              skip_layernorm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm, layer_norm, skip_layernorm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_out, layer_norm_out,
                              skip_layernorm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_bias, layer_norm_bias,
                              skip_layernorm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_scale, layer_norm_scale,
                              skip_layernorm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_mean, layer_norm_mean,
                              skip_layernorm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_variance, layer_norm_variance,
                              skip_layernorm_pattern);
    end_pattern_elt_out.push_back(eltwise_add_out);
    std::unordered_set<Node*> rm_nodes;
    rm_nodes.insert({layer_norm, layer_norm_mean, layer_norm_variance});
    end_pattern_remove_nodes.push_back(rm_nodes);
    end_pattern_biases.push_back(layer_norm_bias);
    end_pattern_scales.push_back(layer_norm_scale);
    end_pattern_out.push_back(layer_norm_out);
    end_patter_layernorms.push_back(layer_norm);
  };
  gpd3(graph, handler3);

  if (start_pattern_in_nodes.empty() || end_pattern_elt_out.empty()) {
    return 0;
  }
  // only reserve the subgraphs that in connected domains.
  int fusion_count = 0;
  // fusion_id for (i, k, js)
  std::vector<std::pair<size_t, std::pair<size_t, std::vector<size_t>>>>
      fusion_ids;
  for (size_t i = 0; i < start_pattern_in_nodes.size(); ++i) {
    Node* tmp = start_pattern_out_node[i];
    Node* old_tmp = nullptr;
    // get correct inner pattern node order.
    std::vector<size_t> js;
    while (tmp != old_tmp) {
      old_tmp = tmp;
      for (size_t j = 0; j < inner_pattern_tmp_in.size(); ++j) {
        if (inner_pattern_tmp_in[j] == tmp) {
          tmp = inner_pattern_out[j];
          js.push_back(j);
          break;
        }
      }
    }

    for (size_t k = 0; k < end_pattern_elt_out.size(); ++k) {
      if (tmp == end_pattern_elt_out[k]) {
        fusion_ids.push_back(std::make_pair(i, std::make_pair(k, js)));
        break;
      }
    }
  }

  for (size_t num = 0; num < fusion_ids.size(); ++num) {
    int i = fusion_ids[num].first;
    int k = fusion_ids[num].second.first;
    std::vector<size_t> js = fusion_ids[num].second.second;

    std::vector<std::string> ids;
    std::vector<std::string> embs;
    for (size_t iter = 0; iter < start_pattern_in_nodes[i].size(); ++iter) {
      ids.push_back(start_pattern_in_nodes[i][iter].first->Name());
      embs.push_back(start_pattern_in_nodes[i][iter].second->Name());
    }
    for (size_t iter = 0; iter < js.size(); ++iter) {
      ids.push_back(inner_pattern_ins[js[iter]].first->Name());
      embs.push_back(inner_pattern_ins[js[iter]].second->Name());
    }
    OpDesc new_op_desc;
    new_op_desc.SetType("fused_embedding_eltwise_layernorm");
    new_op_desc.SetInput("Ids", ids);
    new_op_desc.SetInput("Embs", embs);
    new_op_desc.SetInput("Bias", {end_pattern_biases[k]->Name()});
    new_op_desc.SetInput("Scale", {end_pattern_scales[k]->Name()});
    new_op_desc.SetOutput("Out", {end_pattern_out[k]->Name()});
    new_op_desc.SetAttr("epsilon",
                        end_patter_layernorms[k]->Op()->GetAttr("epsilon"));
    auto* embedding_eltwise_layernorm = graph->CreateOpNode(&new_op_desc);

    for (size_t iter = 0; iter < start_pattern_in_nodes[i].size(); ++iter) {
      IR_NODE_LINK_TO(start_pattern_in_nodes[i][iter].first,
                      embedding_eltwise_layernorm);
      IR_NODE_LINK_TO(start_pattern_in_nodes[i][iter].second,
                      embedding_eltwise_layernorm);
    }
    for (size_t iter = 0; iter < js.size(); ++iter) {
      IR_NODE_LINK_TO(inner_pattern_ins[js[iter]].first,
                      embedding_eltwise_layernorm);
      IR_NODE_LINK_TO(inner_pattern_ins[js[iter]].second,
                      embedding_eltwise_layernorm);
    }
    IR_NODE_LINK_TO(end_pattern_biases[k], embedding_eltwise_layernorm);
    IR_NODE_LINK_TO(end_pattern_scales[k], embedding_eltwise_layernorm);
    IR_NODE_LINK_TO(embedding_eltwise_layernorm, end_pattern_out[k]);

    // Remove unneeded nodes.
    std::unordered_set<const Node*> marked_nodes;
    marked_nodes.insert(start_pattern_remove_nodes[i].begin(),
                        start_pattern_remove_nodes[i].end());
    marked_nodes.insert(end_pattern_remove_nodes[k].begin(),
                        end_pattern_remove_nodes[k].end());
    for (size_t iter = 0; iter < js.size(); ++iter) {
      marked_nodes.insert(inner_pattern_remove_nodes[js[iter]].begin(),
                          inner_pattern_remove_nodes[js[iter]].end());
    }
    GraphSafeRemoveNodes(graph, marked_nodes);
    ++fusion_count;
  }

  return fusion_count;
}

}  // namespace patterns

void EmbeddingEltwiseLayerNormFusePass::ApplyImpl(Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);
  int fusion_count = patterns::BuildFusion(graph, name_scope_);
  if (fusion_count > 0) {
    graph->Set(kEmbEltwiseLayernormPass, new bool(true));
  }
  AddStatis(fusion_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(embedding_eltwise_layernorm_fuse_pass,
              paddle::framework::ir::EmbeddingEltwiseLayerNormFusePass);
REGISTER_PASS_CAPABILITY(embedding_eltwise_layernorm_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("lookup_table", 0)
            .EQ("elementweise_add", 0));
