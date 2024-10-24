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

namespace paddle::framework::ir {
class Node;
}  // namespace paddle::framework::ir

namespace paddle::framework::ir::patterns {

static PDNode* create_emb_vars(PDPattern* pattern,
                               const std::string& name,
                               const std::string& arg,
                               bool is_persist = false) {
  std::unordered_set<std::string> embedding_ops{"lookup_table",
                                                "lookup_table_v2"};
  PDNode* node =
      pattern->NewNode(name)->assert_is_ops_input(embedding_ops, arg);
  if (is_persist) return node->assert_is_persistable_var();
  return node;
}
static PDNode* create_emb_out_vars(PDPattern* pattern,
                                   const std::string& name,
                                   const std::string& arg) {
  std::unordered_set<std::string> embedding_ops{"lookup_table",
                                                "lookup_table_v2"};
  PDNode* node = pattern->NewNode(name)
                     ->assert_is_only_output_of_ops(embedding_ops)
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
  std::unordered_set<std::string> embedding_ops{"lookup_table",
                                                "lookup_table_v2"};
  auto* lookup_table1 =
      pattern->NewNode(lookup_table1_repr())->assert_is_ops(embedding_ops);
  auto* lookup_table2 =
      pattern->NewNode(lookup_table2_repr())->assert_is_ops(embedding_ops);
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
  std::unordered_set<std::string> embedding_ops{"lookup_table",
                                                "lookup_table_v2"};
  auto* lookup_table1 =
      pattern->NewNode(lookup_table1_repr())->assert_is_ops(embedding_ops);
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

}  // namespace paddle::framework::ir::patterns
namespace paddle::framework::ir {

int EmbeddingEltwiseLayerNormFusePass::BuildFusion(
    Graph* graph, const std::string& name_scope
    /*const Scope* scope*/) const {
  GraphPatternDetector gpd;
  auto* pattern = gpd.mutable_pattern();

  std::vector<std::vector<std::pair<Node*, Node*>>> start_pattern_in_nodes;
  std::vector<Node*> start_pattern_out_node;
  std::vector<std::unordered_set<Node*>> start_pattern_remove_nodes;

  // Create pattern.
  patterns::Embedding2Eltwise1Pattern start_pattern(pattern,
                                                    name_scope + "/start");
  start_pattern();
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(lookup_table1_x, lookup_table1_x, start_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(lookup_table2_x, lookup_table2_x, start_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(lookup_table1_w, lookup_table1_w, start_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(lookup_table2_w, lookup_table2_w, start_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(lookup_table1, lookup_table1, start_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(lookup_table2, lookup_table2, start_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        lookup_table1_out, lookup_table1_out, start_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        lookup_table2_out, lookup_table2_out, start_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltwise_add, eltwise_add, start_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltwise_add_out, eltwise_add_out, start_pattern);
    if (!IsCompat(subgraph, graph)) {
      LOG(WARNING) << "Pass(Embedding2Eltwise1Pattern) in op compat failed.";
      return;
    }
    std::vector<std::pair<Node*, Node*>> ins;
    ins.emplace_back(lookup_table1_x, lookup_table1_w);
    ins.emplace_back(lookup_table2_x, lookup_table2_w);
    start_pattern_in_nodes.push_back(ins);
    start_pattern_out_node.push_back(eltwise_add_out);

    std::unordered_set<Node*> rm_nodes;
    rm_nodes.insert({lookup_table1,
                     lookup_table2,
                     lookup_table1_out,
                     lookup_table2_out,
                     eltwise_add,
                     eltwise_add_out});
    start_pattern_remove_nodes.push_back(rm_nodes);
  };
  gpd(graph, handler);

  std::vector<std::pair<Node*, Node*>> inner_pattern_ins;
  std::vector<Node*> inner_pattern_tmp_in;
  std::vector<Node*> inner_pattern_out;
  std::vector<std::unordered_set<Node*>> inner_pattern_remove_nodes;

  GraphPatternDetector gpd2;
  auto* pattern2 = gpd2.mutable_pattern();
  patterns::Embedding1Eltwise1Pattern second_pattern(pattern2,
                                                     name_scope + "/second");
  second_pattern();
  auto handler2 = [&](const GraphPatternDetector::subgraph_t& subgraph,
                      Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(lookup_table1_x, lookup_table1_x, second_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(lookup_table1_w, lookup_table1_w, second_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(lookup_table1, lookup_table1, second_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        lookup_table1_out, lookup_table1_out, second_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltwise_add_in, eltwise_add_in, second_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltwise_add, eltwise_add, second_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltwise_add_out, eltwise_add_out, second_pattern);
    if (!IsCompat(subgraph, graph)) {
      LOG(WARNING) << "Pass(Embedding1Eltwise1Pattern) in op compat failed.";
      return;
    }
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
  std::vector<Node*> end_pattern_layernorms;
  std::vector<std::unordered_set<Node*>> end_pattern_remove_nodes;
  GraphPatternDetector gpd3;
  auto* pattern3 = gpd3.mutable_pattern();
  patterns::SkipLayerNorm skip_layernorm_pattern(pattern3,
                                                 name_scope + "/third");
  skip_layernorm_pattern();
  auto handler3 = [&](const GraphPatternDetector::subgraph_t& subgraph,
                      Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(eltwise_add, eltwise_add, skip_layernorm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        eltwise_add_out, eltwise_add_out, skip_layernorm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm, layer_norm, skip_layernorm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm_out, layer_norm_out, skip_layernorm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm_bias, layer_norm_bias, skip_layernorm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm_scale, layer_norm_scale, skip_layernorm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm_mean, layer_norm_mean, skip_layernorm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm_variance, layer_norm_variance, skip_layernorm_pattern);
    if (!IsCompat(subgraph, graph)) {
      LOG(WARNING) << "Pass(SkipLayerNorm) in op compat failed.";
      return;
    }
    end_pattern_elt_out.push_back(eltwise_add_out);
    std::unordered_set<Node*> rm_nodes;
    rm_nodes.insert({layer_norm, layer_norm_mean, layer_norm_variance});
    end_pattern_remove_nodes.push_back(rm_nodes);
    end_pattern_biases.push_back(layer_norm_bias);
    end_pattern_scales.push_back(layer_norm_scale);
    end_pattern_out.push_back(layer_norm_out);
    end_pattern_layernorms.push_back(layer_norm);
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
        fusion_ids.emplace_back(i, std::make_pair(k, js));
        break;
      }
    }
  }

  for (auto& fusion_id : fusion_ids) {
    int i = fusion_id.first;
    int k = fusion_id.second.first;
    std::vector<size_t> js = fusion_id.second.second;

    std::vector<std::string> ids;
    std::vector<std::string> embs;

    auto ids0_shape = start_pattern_in_nodes[i][0].first->Var()->GetShape();
    bool flag = true;
    for (auto& start_pattern : start_pattern_in_nodes[i]) {
      auto ids_shape = start_pattern.first->Var()->GetShape();
      if (ids_shape.size() != ids0_shape.size()) {
        VLOG(3) << "Shape check failed, ids'rank are not all equal, stop "
                   "embedding_eltwise_layernorm_fuse_pass.";
        flag = false;
      } else {
        for (size_t j = 0; j < ids_shape.size(); ++j) {
          if (ids_shape[j] != ids0_shape[j]) {
            VLOG(3)
                << "Shape check failed, ids.shape[i] are not all equal, stop "
                   "embedding_eltwise_layernorm_fuse_pass.";
            flag = false;
          }
        }
      }
      ids.push_back(start_pattern.first->Name());
      embs.push_back(start_pattern.second->Name());
    }
    for (auto item : js) {
      auto ids_shape = inner_pattern_ins[item].first->Var()->GetShape();
      if (ids_shape.size() != ids0_shape.size()) {
        VLOG(3) << "Shape check failed, ids'rank are not all equal, stop "
                   "embedding_eltwise_layernorm_fuse_pass.";
        flag = false;
      } else {
        for (size_t j = 0; j < ids_shape.size(); ++j) {
          if (ids_shape[j] != ids0_shape[j]) {
            VLOG(3)
                << "Shape check failed, ids.shape[i] are not all equal, stop "
                   "embedding_eltwise_layernorm_fuse_pass.";
            flag = false;
          }
        }
      }
      ids.push_back(inner_pattern_ins[item].first->Name());
      embs.push_back(inner_pattern_ins[item].second->Name());
    }

    // todo: support any inputs with lookup_table_v2
    if (ids.size() < 3) {
      VLOG(3) << "embedding_eltwise_layernorm_fuse_pass only support >=3 "
                 "inputs with lookup_table_v2";
      return fusion_count;
    }
    if (flag) {
      OpDesc new_op_desc;
      new_op_desc.SetType("fused_embedding_eltwise_layernorm");
      new_op_desc.SetInput("Ids", ids);
      new_op_desc.SetInput("Embs", embs);
      new_op_desc.SetInput("WordId", {ids[0]});
      new_op_desc.SetInput("PosId", {ids[1]});
      if (ids.size() > 2) {
        new_op_desc.SetInput("SentId", {ids[2]});
      }

      new_op_desc.SetInput("WordEmbedding", {embs[0]});
      new_op_desc.SetInput("PosEmbedding", {embs[1]});
      if (embs.size() > 2) {
        new_op_desc.SetInput("SentEmbedding", {embs[2]});
      }

      new_op_desc.SetInput("Bias", {end_pattern_biases[k]->Name()});
      new_op_desc.SetInput("Scale", {end_pattern_scales[k]->Name()});
      new_op_desc.SetOutput("Out", {end_pattern_out[k]->Name()});
      new_op_desc.SetAttr("epsilon",
                          end_pattern_layernorms[k]->Op()->GetAttr("epsilon"));

      if (end_pattern_layernorms[k]->Op()->HasAttr("out_threshold")) {
        new_op_desc.SetAttr("enable_int8", true);
        new_op_desc.SetAttr(
            "out_threshold",
            end_pattern_layernorms[k]->Op()->GetAttr("out_threshold"));
      }

      auto* embedding_eltwise_layernorm = graph->CreateOpNode(&new_op_desc);

      for (auto& start_pattern : start_pattern_in_nodes[i]) {
        IR_NODE_LINK_TO(start_pattern.first, embedding_eltwise_layernorm);
        IR_NODE_LINK_TO(start_pattern.second, embedding_eltwise_layernorm);
      }
      for (auto item : js) {
        IR_NODE_LINK_TO(inner_pattern_ins[item].first,
                        embedding_eltwise_layernorm);
        IR_NODE_LINK_TO(inner_pattern_ins[item].second,
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
      for (auto item : js) {
        marked_nodes.insert(inner_pattern_remove_nodes[item].begin(),
                            inner_pattern_remove_nodes[item].end());
      }
      GraphSafeRemoveNodes(graph, marked_nodes);
      ++fusion_count;
    } else {
      VLOG(3) << "Shape check failed, stop "
                 "embedding_eltwise_layernorm_fuse_pass.";
    }
  }

  return fusion_count;
}

EmbeddingEltwiseLayerNormFusePass::EmbeddingEltwiseLayerNormFusePass() {
  AddOpCompat(OpCompat("elementwise_add"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .End();

  AddOpCompat(OpCompat("layer_norm"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Scale")
      .IsTensor()
      .End()
      .AddInput("Bias")
      .IsTensor()
      .End()
      .AddOutput("Y")
      .IsTensor()
      .End()
      .AddOutput("Mean")
      .IsTensor()
      .End()
      .AddOutput("Variance")
      .IsTensor()
      .End()
      .AddAttr("epsilon")
      .IsNumGE(0.0f)
      .IsNumLE(0.001f)
      .End()
      .AddAttr("begin_norm_axis")
      .IsNumGT(0)
      .End();
}

void EmbeddingEltwiseLayerNormFusePass::ApplyImpl(Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);
  int fusion_count =
      EmbeddingEltwiseLayerNormFusePass::BuildFusion(graph, name_scope_);
  if (fusion_count > 0) {
    graph->Set(kEmbEltwiseLayernormPass, new bool(true));
  }
  AddStatis(fusion_count);
}

}  // namespace paddle::framework::ir

REGISTER_PASS(embedding_eltwise_layernorm_fuse_pass,
              paddle::framework::ir::EmbeddingEltwiseLayerNormFusePass);
REGISTER_PASS_CAPABILITY(embedding_eltwise_layernorm_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("lookup_table", 1)
            .LE("lookup_table_v2", 1)
            .LE("elementweise_add", 1));
