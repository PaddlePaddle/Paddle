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
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/lod_tensor.h"

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

static int BuildFusion(Graph* graph, const std::string& name_scope,
                       const Scope* scope) {
  GraphPatternDetector gpd;
  auto* pattern = gpd.mutable_pattern();

  // Create pattern.
  EmbeddingEltwiseLayerNormPattern emb_eltwise_layernorm_pattern(pattern,
                                                                 name_scope);
  emb_eltwise_layernorm_pattern();

  int fusion_count{0};
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(lookup_table1_x, lookup_table1_x,
                              emb_eltwise_layernorm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(lookup_table2_x, lookup_table2_x,
                              emb_eltwise_layernorm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(lookup_table3_x, lookup_table3_x,
                              emb_eltwise_layernorm_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(lookup_table1_w, lookup_table1_w,
                              emb_eltwise_layernorm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(lookup_table2_w, lookup_table2_w,
                              emb_eltwise_layernorm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(lookup_table3_w, lookup_table3_w,
                              emb_eltwise_layernorm_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(lookup_table1, lookup_table1,
                              emb_eltwise_layernorm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(lookup_table2, lookup_table2,
                              emb_eltwise_layernorm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(lookup_table3, lookup_table3,
                              emb_eltwise_layernorm_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(lookup_table1_out, lookup_table1_out,
                              emb_eltwise_layernorm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(lookup_table2_out, lookup_table2_out,
                              emb_eltwise_layernorm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(lookup_table3_out, lookup_table3_out,
                              emb_eltwise_layernorm_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(eltwise_add_12, eltwise_add_12,
                              emb_eltwise_layernorm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltwise_add_12_out, eltwise_add_12_out,
                              emb_eltwise_layernorm_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(eltwise_add, eltwise_add,
                              emb_eltwise_layernorm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltwise_add_out, eltwise_add_out,
                              emb_eltwise_layernorm_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(layer_norm, layer_norm,
                              emb_eltwise_layernorm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_out, layer_norm_out,
                              emb_eltwise_layernorm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_bias, layer_norm_bias,
                              emb_eltwise_layernorm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_scale, layer_norm_scale,
                              emb_eltwise_layernorm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_mean, layer_norm_mean,
                              emb_eltwise_layernorm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_variance, layer_norm_variance,
                              emb_eltwise_layernorm_pattern);

    auto get_persist_tensor_dims = [&](std::string name) -> framework::DDim {
      auto* var = scope->FindVar(name);
      PADDLE_ENFORCE_NOT_NULL(var,
                              platform::errors::PreconditionNotMet(
                                  "Cant not found the %d var in scope.", name));
      return var->GetMutable<LoDTensor>()->dims();
    };

    // Check the weight dims.
    auto word_emb_dims = get_persist_tensor_dims(lookup_table1_w->Name());
    auto pos_emb_dims = get_persist_tensor_dims(lookup_table2_w->Name());
    auto sent_emb_dims = get_persist_tensor_dims(lookup_table3_w->Name());
    if (word_emb_dims.size() != 2 || pos_emb_dims.size() != 2 ||
        sent_emb_dims.size() != 2 || word_emb_dims[1] != pos_emb_dims[1] ||
        word_emb_dims[1] != sent_emb_dims[1]) {
      return;
    }

    OpDesc new_op_desc;
    new_op_desc.SetType("fused_embedding_eltwise_layernorm");
    new_op_desc.SetInput("WordId", {lookup_table1_x->Name()});
    new_op_desc.SetInput("PosId", {lookup_table2_x->Name()});
    new_op_desc.SetInput("SentId", {lookup_table3_x->Name()});

    new_op_desc.SetInput("WordEmb", {lookup_table1_w->Name()});
    new_op_desc.SetInput("PosEmb", {lookup_table2_w->Name()});
    new_op_desc.SetInput("SentEmb", {lookup_table3_w->Name()});

    new_op_desc.SetInput("Bias", {layer_norm_bias->Name()});
    new_op_desc.SetInput("Scale", {layer_norm_scale->Name()});
    new_op_desc.SetOutput("Out", {layer_norm_out->Name()});
    new_op_desc.SetAttr("epsilon", layer_norm->Op()->GetAttr("epsilon"));

    auto* embedding_eltwise_layernorm = graph->CreateOpNode(&new_op_desc);
    IR_NODE_LINK_TO(lookup_table1_x, embedding_eltwise_layernorm);
    IR_NODE_LINK_TO(lookup_table2_x, embedding_eltwise_layernorm);
    IR_NODE_LINK_TO(lookup_table3_x, embedding_eltwise_layernorm);

    IR_NODE_LINK_TO(lookup_table1_w, embedding_eltwise_layernorm);
    IR_NODE_LINK_TO(lookup_table2_w, embedding_eltwise_layernorm);
    IR_NODE_LINK_TO(lookup_table3_w, embedding_eltwise_layernorm);
    IR_NODE_LINK_TO(layer_norm_bias, embedding_eltwise_layernorm);
    IR_NODE_LINK_TO(layer_norm_scale, embedding_eltwise_layernorm);
    IR_NODE_LINK_TO(embedding_eltwise_layernorm, layer_norm_out);

    std::unordered_set<const Node*> marked_nodes(
        {lookup_table1, lookup_table2, lookup_table3, lookup_table1_out,
         lookup_table2_out, lookup_table3_out, eltwise_add_12,
         eltwise_add_12_out, eltwise_add, eltwise_add_out, layer_norm,
         layer_norm_mean, layer_norm_variance});
    // Remove unneeded nodes.
    GraphSafeRemoveNodes(graph, marked_nodes);
    ++fusion_count;
  };
  gpd(graph, handler);

  return fusion_count;
}

PDNode* EmbeddingEltwiseLayerNormPattern::operator()() {
  // Create shared nodes.
  auto create_emb_vars = [&](const std::string& name, const std::string& arg,
                             bool is_persist = false) -> PDNode* {
    PDNode* node = pattern->NewNode(name)
                       ->assert_is_op_input("lookup_table", arg)
                       ->AsInput();
    if (is_persist) return node->assert_is_persistable_var();
    return node;
  };

  auto create_emb_out_vars = [&](const std::string& name,
                                 const std::string& arg) -> PDNode* {
    PDNode* node = pattern->NewNode(name)
                       ->AsIntermediate()
                       ->assert_is_op_output("lookup_table")
                       ->assert_is_op_input("elementwise_add", arg);
    return node;
  };

  auto* lookup_table1_x = create_emb_vars(lookup_table1_x_repr(), "Ids");
  auto* lookup_table2_x = create_emb_vars(lookup_table2_x_repr(), "Ids");
  auto* lookup_table3_x = create_emb_vars(lookup_table3_x_repr(), "Ids");
  auto* lookup_table1_w = create_emb_vars(lookup_table1_w_repr(), "W", true);
  auto* lookup_table2_w = create_emb_vars(lookup_table2_w_repr(), "W", true);
  auto* lookup_table3_w = create_emb_vars(lookup_table3_w_repr(), "W", true);

  auto* lookup_table1 =
      pattern->NewNode(lookup_table1_repr())->assert_is_op("lookup_table");
  auto* lookup_table2 =
      pattern->NewNode(lookup_table2_repr())->assert_is_op("lookup_table");
  auto* lookup_table3 =
      pattern->NewNode(lookup_table3_repr())->assert_is_op("lookup_table");

  auto* lookup_table1_out = create_emb_out_vars(lookup_table1_out_repr(), "X");
  auto* lookup_table2_out = create_emb_out_vars(lookup_table2_out_repr(), "Y");
  auto* lookup_table3_out = create_emb_out_vars(lookup_table3_out_repr(), "Y");

  auto* eltwise_add_12 =
      pattern->NewNode(eltwise_add_12_repr())->assert_is_op("elementwise_add");
  auto* eltwise_add_12_out = pattern->NewNode(eltwise_add_12_out_repr())
                                 ->AsIntermediate()
                                 ->assert_is_op_output("elementwise_add")
                                 ->assert_is_op_input("elementwise_add", "X");

  auto* eltwise_add =
      pattern->NewNode(eltwise_add_repr())->assert_is_op("elementwise_add");
  auto* eltwise_add_out = pattern->NewNode(eltwise_add_out_repr())
                              ->AsIntermediate()
                              ->assert_is_op_output("elementwise_add");

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

  // Link all nodes together
  lookup_table1->LinksFrom({lookup_table1_x, lookup_table1_w})
      .LinksTo({lookup_table1_out});
  lookup_table2->LinksFrom({lookup_table2_x, lookup_table2_w})
      .LinksTo({lookup_table2_out});
  lookup_table3->LinksFrom({lookup_table3_x, lookup_table3_w})
      .LinksTo({lookup_table3_out});
  eltwise_add_12->LinksFrom({lookup_table1_out, lookup_table2_out})
      .LinksTo({eltwise_add_12_out});
  eltwise_add->LinksFrom({lookup_table3_out, eltwise_add_12_out})
      .LinksTo({eltwise_add_out});
  layer_norm
      ->LinksFrom({eltwise_add_out, layer_norm_bias_var, layer_norm_scale_var})
      .LinksTo({layer_norm_out, layer_norm_mean_var, layer_norm_variance_var});
  return layer_norm_out;
}

}  // namespace patterns

void EmbeddingEltwiseLayerNormFusePass::ApplyImpl(Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);
  auto* scope = param_scope();
  PADDLE_ENFORCE_NOT_NULL(
      scope, platform::errors::PreconditionNotMet(
                 "The scope is null, please initialize the scope first."));
  int fusion_count = patterns::BuildFusion(graph, name_scope_, scope);
  AddStatis(fusion_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(embedding_eltwise_layernorm_fuse_pass,
              paddle::framework::ir::EmbeddingEltwiseLayerNormFusePass);
