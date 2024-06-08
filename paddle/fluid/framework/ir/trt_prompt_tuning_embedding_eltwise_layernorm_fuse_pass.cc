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

#include "paddle/fluid/framework/ir/trt_prompt_tuning_embedding_eltwise_layernorm_fuse_pass.h"

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
void TrtPromptTuningEmbedding2Eltwise1Pattern::operator()() {
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
void TrtPromptTuningEmbedding1Eltwise1Pattern::operator()() {
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
void TrtPromptTuningSkipLayerNorm::operator()() {
  auto* eltwise_add =
      pattern->NewNode(eltwise_add_repr())->assert_is_op("elementwise_add");
  auto* eltwise_add_out = pattern->NewNode(eltwise_add_out_repr())
                              ->assert_is_op_output("elementwise_add")
                              ->AsIntermediate();

  auto* mul0_x = pattern->NewNode(mul0_x_repr())
                     ->assert_is_op_input("matrix_multiply", "X");

  auto* mul0_y = pattern->NewNode(mul0_y_repr())
                     ->assert_is_op_input("matrix_multiply", "Y");

  auto* mul0 = pattern->NewNode(mul0_repr())->assert_is_op("matrix_multiply");

  auto* mul0_out = pattern->NewNode(mul0_out_repr())
                       ->assert_is_op_output("matrix_multiply")
                       ->assert_is_op_input("elementwise_add", "X")
                       ->AsIntermediate();

  auto* eltadd0_b = pattern->NewNode(eltadd0_b_repr())
                        ->assert_is_op_input("elementwise_add", "Y");

  auto* eltadd0 =
      pattern->NewNode(eltadd0_repr())->assert_is_op("elementwise_add");

  auto* eltadd0_out = pattern->NewNode(eltadd0_out_repr())
                          ->assert_is_op_output("elementwise_add")
                          ->assert_is_op_input("relu")
                          ->AsIntermediate();

  auto* relu = pattern->NewNode(relu_repr())->assert_is_op("relu");
  auto* relu_out = pattern->NewNode(relu_out_repr())
                       ->assert_is_op_output("relu")
                       ->assert_is_op_input("matrix_multiply", "X")
                       ->AsIntermediate();

  auto* mul1_y = pattern->NewNode(mul1_y_repr())
                     ->assert_is_op_input("matrix_multiply", "Y");
  auto* mul1 = pattern->NewNode(mul1_repr())->assert_is_op("matrix_multiply");

  auto* mul1_out = pattern->NewNode(mul1_out_repr())
                       ->assert_is_op_output("matrix_multiply")
                       ->assert_is_op_input("elementwise_add", "X")
                       ->AsIntermediate();

  auto* eltadd1_b = pattern->NewNode(eltadd1_b_repr())
                        ->assert_is_op_input("elementwise_add", "Y");

  auto* eltadd1 =
      pattern->NewNode(eltadd1_repr())->assert_is_op("elementwise_add");

  auto* eltadd1_out = pattern->NewNode(eltadd1_out_repr())
                          ->assert_is_op_output("elementwise_add");

  auto* concat = pattern->NewNode(concat_repr())->assert_is_op("concat");

  auto* concat_out = pattern->NewNode(concat_out_repr())
                         ->assert_is_op_output("concat")
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

  eltwise_add->LinksTo({eltwise_add_out});

  mul0->LinksFrom({mul0_x, mul0_y}).LinksTo({mul0_out});

  eltadd0->LinksFrom({mul0_out, eltadd0_b}).LinksTo({eltadd0_out});

  relu->LinksFrom({eltadd0_out}).LinksTo({relu_out});

  mul1->LinksFrom({relu_out, mul1_y}).LinksTo({mul1_out});

  eltadd1->LinksFrom({mul1_out, eltadd1_b}).LinksTo({eltadd1_out});

  concat->LinksFrom({eltadd1_out, eltwise_add_out}).LinksTo({concat_out});

  layer_norm->LinksFrom({concat_out, layer_norm_bias_var, layer_norm_scale_var})
      .LinksTo({layer_norm_out});
}

}  // namespace paddle::framework::ir::patterns
namespace paddle::framework::ir {

int TrtPromptTuningEmbeddingEltwiseLayerNormFusePass::BuildFusion(
    Graph* graph, const std::string& name_scope
    /*const Scope* scope*/) const {
  GraphPatternDetector gpd;
  auto* pattern = gpd.mutable_pattern();
  std::string pos_id = Get<std::string>("tensorrt_transformer_posid");
  std::string mask_id = Get<std::string>("tensorrt_transformer_maskid");
  std::vector<std::vector<std::pair<Node*, Node*>>> start_pattern_in_nodes;
  std::vector<Node*> start_pattern_out_node;
  std::vector<std::unordered_set<Node*>> start_pattern_remove_nodes;

  // Create pattern.
  patterns::TrtPromptTuningEmbedding2Eltwise1Pattern start_pattern(
      pattern, name_scope + "/start");
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
      LOG(WARNING) << "Pass(TrtPromptTuningEmbedding2Eltwise1Pattern) in op "
                      "compat failed.";
      return;
    }
    std::vector<std::pair<Node*, Node*>> ins;
    ins.push_back(std::make_pair(lookup_table1_x, lookup_table1_w));
    ins.push_back(std::make_pair(lookup_table2_x, lookup_table2_w));
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
  patterns::TrtPromptTuningEmbedding1Eltwise1Pattern second_pattern(
      pattern2, name_scope + "/second");
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
      LOG(WARNING) << "Pass(TrtPromptTuningEmbedding1Eltwise1Pattern) in op "
                      "compat failed.";
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
  std::vector<Node*> end_pattern_eltadd1;
  std::vector<Node*> end_pattern_eltadd1_out;
  std::vector<Node*> end_pattern_concat;
  std::vector<Node*> end_pattern_concat_out;
  std::vector<Node*> end_pattern_scales;
  std::vector<Node*> end_pattern_biases;
  std::vector<Node*> end_pattern_out;
  std::vector<Node*> end_pattern_layernorms;
  std::vector<std::unordered_set<Node*>> end_pattern_remove_nodes;
  GraphPatternDetector gpd3;
  auto* pattern3 = gpd3.mutable_pattern();
  patterns::TrtPromptTuningSkipLayerNorm skip_layernorm_pattern(
      pattern3, name_scope + "/third");
  skip_layernorm_pattern();
  auto handler3 = [&](const GraphPatternDetector::subgraph_t& subgraph,
                      Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(eltwise_add, eltwise_add, skip_layernorm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        eltwise_add_out, eltwise_add_out, skip_layernorm_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(eltadd1, eltadd1, skip_layernorm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd1_out, eltadd1_out, skip_layernorm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(concat, concat, skip_layernorm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(concat_out, concat_out, skip_layernorm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm, layer_norm, skip_layernorm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm_out, layer_norm_out, skip_layernorm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm_bias, layer_norm_bias, skip_layernorm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm_scale, layer_norm_scale, skip_layernorm_pattern);
    if (!IsCompat(subgraph, graph)) {
      LOG(WARNING) << "Pass(TrtPromptTuningSkipLayerNorm) in op compat failed.";
      return;
    }
    end_pattern_elt_out.push_back(eltwise_add_out);
    std::unordered_set<Node*> rm_nodes;
    rm_nodes.insert({concat});
    rm_nodes.insert({concat_out});
    rm_nodes.insert({layer_norm});
    end_pattern_remove_nodes.push_back(rm_nodes);

    end_pattern_eltadd1.push_back(eltadd1);
    end_pattern_eltadd1_out.push_back(eltadd1_out);
    end_pattern_concat.push_back(concat);
    end_pattern_concat_out.push_back(concat_out);
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
        fusion_ids.push_back(std::make_pair(i, std::make_pair(k, js)));
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
    for (auto& item : start_pattern_in_nodes[i]) {
      auto ids_shape = item.first->Var()->GetShape();
      if (ids_shape.size() != ids0_shape.size()) {
        VLOG(3) << "Shape check failed, ids'rank are not all equal, stop "
                   "trt_prompt_tuning_embedding_eltwise_layernorm_fuse_pass.";
        flag = false;
      } else {
        for (size_t j = 0; j < ids_shape.size(); ++j) {
          if (ids_shape[j] != ids0_shape[j]) {
            VLOG(3)
                << "Shape check failed, ids.shape[i] are not all equal, stop "
                   "trt_prompt_tuning_embedding_eltwise_layernorm_fuse_pass.";
            flag = false;
          }
        }
      }
      ids.push_back(item.first->Name());
      embs.push_back(item.second->Name());
    }
    for (auto item : js) {
      auto ids_shape = inner_pattern_ins[item].first->Var()->GetShape();
      if (ids_shape.size() != ids0_shape.size()) {
        VLOG(3) << "Shape check failed, ids'rank are not all equal, stop "
                   "trt_prompt_tuning_embedding_eltwise_layernorm_fuse_pass.";
        flag = false;
      } else {
        for (size_t j = 0; j < ids_shape.size(); ++j) {
          if (ids_shape[j] != ids0_shape[j]) {
            VLOG(3)
                << "Shape check failed, ids.shape[i] are not all equal, stop "
                   "trt_prompt_tuning_embedding_eltwise_layernorm_fuse_pass.";
            flag = false;
          }
        }
      }
      ids.push_back(inner_pattern_ins[item].first->Name());
      embs.push_back(inner_pattern_ins[item].second->Name());
    }

    if (flag) {
      OpDesc new_op_desc(end_pattern_layernorms[0]->Op()->Block());
      new_op_desc.SetType("prompt_tuning_emb_eltwise_layernorm");
      new_op_desc.SetInput("Ids", ids);
      new_op_desc.SetInput("Embs", embs);
      new_op_desc.SetInput("PosId", {pos_id});
      new_op_desc.SetInput("MaskId", {mask_id});

      new_op_desc.SetInput("Bias", {end_pattern_biases[k]->Name()});
      new_op_desc.SetInput("Scale", {end_pattern_scales[k]->Name()});
      new_op_desc.SetInput("DenseVector", {end_pattern_eltadd1_out[k]->Name()});
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

      for (auto& item : start_pattern_in_nodes[i]) {
        IR_NODE_LINK_TO(item.first, embedding_eltwise_layernorm);
        IR_NODE_LINK_TO(item.second, embedding_eltwise_layernorm);
      }
      for (auto item : js) {
        IR_NODE_LINK_TO(inner_pattern_ins[item].first,
                        embedding_eltwise_layernorm);
        IR_NODE_LINK_TO(inner_pattern_ins[item].second,
                        embedding_eltwise_layernorm);
      }
      IR_NODE_LINK_TO(end_pattern_biases[k], embedding_eltwise_layernorm);
      IR_NODE_LINK_TO(end_pattern_scales[k], embedding_eltwise_layernorm);
      IR_NODE_LINK_TO(end_pattern_eltadd1_out[k], embedding_eltwise_layernorm);
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
                 "trt_prompt_tuning_embedding_eltwise_layernorm_fuse_pass.";
    }
  }
  return fusion_count;
}

TrtPromptTuningEmbeddingEltwiseLayerNormFusePass::
    TrtPromptTuningEmbeddingEltwiseLayerNormFusePass() {
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

  AddOpCompat(OpCompat("relu"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End();

  AddOpCompat(OpCompat("concat"))
      .AddInput("X")
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

void TrtPromptTuningEmbeddingEltwiseLayerNormFusePass::ApplyImpl(
    Graph* graph) const {
  bool with_dynamic_shape = Get<bool>("with_dynamic_shape");
  if (!with_dynamic_shape) {
    VLOG(3) << "Stop this pass, because "
               "trt_prompt_tuning_embedding_eltwise_layernorm_fuse_pass need: "
               "use_varseqlen, "
               "with_dynamic_shape."
               "please reconfig.";
    return;
  }
  FusePassBase::Init(name_scope_, graph);
  int fusion_count =
      TrtPromptTuningEmbeddingEltwiseLayerNormFusePass::BuildFusion(
          graph, name_scope_);
  if (fusion_count > 0) {
    bool use_varseqlen = Get<bool>("use_varseqlen");
    std::string pos_id = Get<std::string>("tensorrt_transformer_posid");
    std::string mask_id = Get<std::string>("tensorrt_transformer_maskid");

    if ((use_varseqlen && !pos_id.empty() && !mask_id.empty())) {
      VLOG(3)
          << "start trt_prompt_tuning_embedding_eltwise_layernorm_fuse_pass";
    } else {
      VLOG(3) << "Stop this pass, because "
                 "trt_prompt_tuning_embedding_eltwise_layernorm_fuse_pass only "
                 "support use_varseqlen, please reconfig";
      return;
    }
    graph->Set(kEmbEltwiseLayernormPass, new bool(true));
  }
  AddStatis(fusion_count);
}

}  // namespace paddle::framework::ir

REGISTER_PASS(
    trt_prompt_tuning_embedding_eltwise_layernorm_fuse_pass,
    paddle::framework::ir::TrtPromptTuningEmbeddingEltwiseLayerNormFusePass);
REGISTER_PASS_CAPABILITY(
    trt_prompt_tuning_embedding_eltwise_layernorm_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("lookup_table", 1)
            .LE("lookup_table_v2", 1)
            .LE("elementweise_add", 1));
