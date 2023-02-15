// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/remove_padding_recover_padding_pass.h"

#include <string>

#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {
void EmbEltwiseLayernorm::operator()() {
  // Create nodes for fused_embedding_eltwise_layernorm.
  auto* emb_elt_layernorm_op =
      pattern->NewNode(emb_elt_layernorm_op_repr())
          ->assert_is_op("fused_embedding_eltwise_layernorm");
  auto* emb_elt_layernorm_out =
      pattern->NewNode(emb_elt_layernorm_out_repr())
          ->assert_is_op_output("fused_embedding_eltwise_layernorm", "Out");

  // Add links for fused_embedding_eltwise_layernorm op.
  emb_elt_layernorm_op->LinksTo({emb_elt_layernorm_out});
}

void PrelnEmbEltwiseLayernorm::operator()() {
  // Create nodes for fused_preln_embedding_eltwise_layernorm.
  auto* preln_emb_elt_layernorm_op =
      pattern->NewNode(preln_emb_elt_layernorm_op_repr())
          ->assert_is_op("fused_preln_embedding_eltwise_layernorm");
  auto* preln_emb_elt_layernorm_out_0 =
      pattern->NewNode(preln_emb_elt_layernorm_out_0_repr())
          ->assert_is_op_output("fused_preln_embedding_eltwise_layernorm",
                                "Out_0");
  auto* preln_emb_elt_layernorm_out_1 =
      pattern->NewNode(preln_emb_elt_layernorm_out_1_repr())
          ->assert_is_op_output("fused_preln_embedding_eltwise_layernorm",
                                "Out_1");

  // Add links for fused_preln_embedding_eltwise_layernorm op.
  preln_emb_elt_layernorm_op->LinksTo(
      {preln_emb_elt_layernorm_out_0, preln_emb_elt_layernorm_out_1});
}

void SkipLayernorm::operator()() {
  // Create nodes for skip_layernorm.
  auto* skip_layernorm_x = pattern->NewNode(skip_layernorm_x_repr())
                               ->assert_is_op_input("skip_layernorm", "X");
  auto* skip_layernorm_y = pattern->NewNode(skip_layernorm_y_repr())
                               ->assert_is_op_input("skip_layernorm", "Y");
  auto* skip_layernorm_op = pattern->NewNode(skip_layernorm_op_repr())
                                ->assert_is_op("skip_layernorm");
  auto* skip_layernorm_out = pattern->NewNode(skip_layernorm_out_repr())
                                 ->assert_is_op_output("skip_layernorm", "Out");

  // Add links for skip_layernorm op.
  skip_layernorm_op->LinksFrom({skip_layernorm_x, skip_layernorm_y})
      .LinksTo({skip_layernorm_out});
}

void PrelnSkipLayernorm::operator()() {
  // Create nodes for preln_skip_layernorm.
  auto* preln_skip_layernorm_x =
      pattern->NewNode(preln_skip_layernorm_x_repr())
          ->assert_is_op_input("preln_skip_layernorm", "X");
  auto* preln_skip_layernorm_y =
      pattern->NewNode(preln_skip_layernorm_y_repr())
          ->assert_is_op_input("preln_skip_layernorm", "Y");
  auto* preln_skip_layernorm_op =
      pattern->NewNode(preln_skip_layernorm_op_repr())
          ->assert_is_op("preln_skip_layernorm");
  auto* preln_skip_layernorm_out_0 =
      pattern->NewNode(preln_skip_layernorm_out_0_repr())
          ->assert_is_op_output("preln_skip_layernorm", "Out_0");
  auto* preln_skip_layernorm_out_1 =
      pattern->NewNode(preln_skip_layernorm_out_1_repr())
          ->assert_is_op_output("preln_skip_layernorm", "Out_1");

  // Add links for preln_skip_layernorm op.
  preln_skip_layernorm_op
      ->LinksFrom({preln_skip_layernorm_x, preln_skip_layernorm_y})
      .LinksTo({preln_skip_layernorm_out_0, preln_skip_layernorm_out_1});
}

void MultiheadMatmul::operator()() {
  // Create nodes for multihead_matmul.
  auto* multihead_matmul_input =
      pattern->NewNode(multihead_matmul_input_repr())
          ->assert_is_op_input("multihead_matmul", "Input");
  auto* multihead_matmul_op = pattern->NewNode(multihead_matmul_op_repr())
                                  ->assert_is_op("multihead_matmul");
  auto* multihead_matmul_out =
      pattern->NewNode(multihead_matmul_out_repr())
          ->assert_is_op_output("multihead_matmul", "Out");

  // Add links for multihead_matmul op.
  multihead_matmul_op->LinksFrom({multihead_matmul_input})
      .LinksTo({multihead_matmul_out});
}

void Fc::operator()() {
  // Create nodes for fc.
  auto* fc_input =
      pattern->NewNode(fc_input_repr())->assert_is_op_input("fc", "Input");
  auto* fc_op = pattern->NewNode(fc_op_repr())->assert_is_op("fc");
  fc_op->LinksFrom({fc_input});
}

void Activation::operator()() {
  // Create nodes for activation.
  std::unordered_set<std::string> activation_ops{"relu", "sigmoid", "gelu"};
  auto* activation_input = pattern->NewNode(activation_input_repr())
                               ->assert_is_ops_input(activation_ops);
  auto* activation_op =
      pattern->NewNode(activation_op_repr())->assert_is_ops(activation_ops);
  auto* activation_out = pattern->NewNode(activation_out_repr())
                             ->assert_is_ops_output(activation_ops);

  // Add links for activation op.
  activation_op->LinksFrom({activation_input}).LinksTo({activation_out});
}

void FusedTokenPrune::operator()() {
  // Create nodes for fused_token_prune.
  auto* fused_token_prune_input =
      pattern->NewNode(fused_token_prune_input_repr())
          ->assert_is_op_input("fused_token_prune", "X");
  auto* fused_token_prune_op = pattern->NewNode(fused_token_prune_op_repr())
                                   ->assert_is_op("fused_token_prune");
  auto* fused_token_prune_output =
      pattern->NewNode(fused_token_prune_output_repr())
          ->assert_is_op_output("fused_token_prune", "SlimmedX");

  fused_token_prune_op->LinksFrom({fused_token_prune_input})
      .LinksTo({fused_token_prune_output});
}
}  // namespace patterns

void RemovePaddingRecoverPaddingPass::ApplyImpl(ir::Graph* graph) const {
  bool use_varseqlen = Get<bool>("use_varseqlen");
  std::string pos_id = Get<std::string>("tensorrt_transformer_posid");
  std::string mask_id = Get<std::string>("tensorrt_transformer_maskid");

  if (use_varseqlen && pos_id != "" && mask_id != "" &&
      (graph->Has(framework::ir::kEmbEltwiseLayernormPass) ||
       graph->Has(framework::ir::kPrelnEmbEltwiseLayernormPass)) &&
      graph->Has(framework::ir::kMultiheadMatmulPass)) {
    VLOG(3) << "start varseqlen remove_padding_recover_padding_pass";
  } else {
    VLOG(3) << "remove_padding_recover_padding_pass check failed";
    return;
  }

  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  FusePassBase::Init(name_scope_, graph);
  auto* scope = param_scope();
  int found_subgraph_count = 0;

  // Create an remove_padding op node
  auto insert_remove_padding_op = [&](Node* input_node, Node* op_node) {
    // create op, var in graph
    OpDesc remove_padding(op_node->Op()->Block());
    std::string remove_padding_out_name =
        input_node->Name() + ".remove_padding";
    auto* remove_padding_out =
        op_node->Op()->Block()->Var(remove_padding_out_name);
    remove_padding_out->SetDataType(input_node->Var()->GetDataType());
    remove_padding_out->SetShape(input_node->Var()->GetShape());
    remove_padding_out->SetPersistable(false);

    // remove_padding_op
    remove_padding.SetType("remove_padding");

    // input
    remove_padding.SetInput("Input", {input_node->Name()});

    // output
    remove_padding.SetOutput("Out", {remove_padding_out_name});

    // set out_threshold for int8
    if (op_node->Op()->HasAttr("Input_scale")) {
      remove_padding.SetAttr("out_threshold",
                             op_node->Op()->GetAttr("Input_scale"));
    } else {
      VLOG(3) << "remove_padding_op has not out_threshold, because next op has "
                 "not Input_scale.";
    }

    auto remove_padding_op_node = graph->CreateOpNode(&remove_padding);
    auto remove_padding_out_node = graph->CreateVarNode(remove_padding_out);

    // replace link
    for (size_t i = 0; i < input_node->outputs.size(); ++i) {
      if (input_node->outputs[i] == op_node) {
        input_node->outputs[i] = remove_padding_op_node;
        remove_padding_op_node->inputs.push_back(input_node);
      }
    }

    // link node
    IR_NODE_LINK_TO(remove_padding_op_node, remove_padding_out_node);

    // replace link
    for (size_t i = 0; i < op_node->inputs.size(); ++i) {
      if (op_node->inputs[i] == input_node) {
        op_node->inputs[i] = remove_padding_out_node;
        remove_padding_out_node->outputs.push_back(op_node);
      }
    }

    // create variable in scope
    scope->Var(remove_padding_out_name);
    auto* remove_padding_out_tensor =
        scope->FindVar(remove_padding_out_name)->GetMutable<phi::DenseTensor>();
    remove_padding_out_tensor->mutable_data<float>(platform::CUDAPlace());

    // rename
    op_node->Op()->RenameInput(input_node->Name(),
                               remove_padding_out_node->Name());
  };

  // create an remove_padding op node
  auto insert_recover_padding_op = [&](Node* op_node, Node* out_node) {
    // create op, var in graph
    OpDesc recover_padding(op_node->Op()->Block());
    std::string recover_padding_input_name =
        out_node->Name() + ".recover_padding";
    auto* recover_padding_input =
        op_node->Op()->Block()->Var(recover_padding_input_name);
    recover_padding_input->SetDataType(out_node->Var()->GetDataType());
    recover_padding_input->SetShape(out_node->Var()->GetShape());
    recover_padding_input->SetPersistable(false);

    // recover_padding_op
    recover_padding.SetType("recover_padding");

    // input
    recover_padding.SetInput("Input", {recover_padding_input_name});

    // output
    recover_padding.SetOutput("Out", {out_node->Name()});

    // set out_threshold for int8
    if (op_node->Op()->HasAttr("out_threshold")) {
      recover_padding.SetAttr("out_threshold",
                              op_node->Op()->GetAttr("out_threshold"));
    } else if (op_node->Op()->HasAttr("out_0_threshold")) {
      recover_padding.SetAttr("out_threshold",
                              op_node->Op()->GetAttr("out_0_threshold"));
    } else if (op_node->Op()->HasAttr("out_1_threshold")) {
      recover_padding.SetAttr("out_threshold",
                              op_node->Op()->GetAttr("out_1_threshold"));
    } else {
      VLOG(3) << "recover_padding_op has not out_threshold, because previous "
                 "op has not out_*_threshold.";
    }

    auto recover_padding_op_node = graph->CreateOpNode(&recover_padding);
    auto recover_padding_input_node =
        graph->CreateVarNode(recover_padding_input);

    // replace link
    for (size_t i = 0; i < op_node->outputs.size(); ++i) {
      if (op_node->outputs[i] == out_node) {
        op_node->outputs[i] = recover_padding_input_node;
        recover_padding_input_node->inputs.push_back(op_node);
      }
    }

    // link node
    IR_NODE_LINK_TO(recover_padding_input_node, recover_padding_op_node);

    // replace link
    for (size_t i = 0; i < out_node->inputs.size(); ++i) {
      if (out_node->inputs[i] == op_node) {
        out_node->inputs[i] = recover_padding_op_node;
        recover_padding_op_node->outputs.push_back(out_node);
      }
    }

    // create variable in scope
    scope->Var(recover_padding_input_name);
    auto* recover_padding_input_tensor =
        scope->FindVar(recover_padding_input_name)
            ->GetMutable<phi::DenseTensor>();
    recover_padding_input_tensor->mutable_data<float>(platform::CUDAPlace());

    // rename
    op_node->Op()->RenameOutput(out_node->Name(), recover_padding_input_name);
  };

  bool check_flag = true;

  GraphPatternDetector gpd0;
  patterns::EmbEltwiseLayernorm fused_embedding_eltwise_layernorm(
      gpd0.mutable_pattern(), "remove_padding_recover_padding_pass");
  fused_embedding_eltwise_layernorm();

  auto handler0 = [&](const GraphPatternDetector::subgraph_t& subgraph,
                      Graph* graph) {
    VLOG(3) << "remove_padding_recover_padding_pass for transformer: "
               "fused_embedding_eltwise_layernorm";

    GET_IR_NODE_FROM_SUBGRAPH(emb_elt_layernorm_op,
                              emb_elt_layernorm_op,
                              fused_embedding_eltwise_layernorm);
    GET_IR_NODE_FROM_SUBGRAPH(emb_elt_layernorm_out,
                              emb_elt_layernorm_out,
                              fused_embedding_eltwise_layernorm);

    insert_recover_padding_op(emb_elt_layernorm_op, emb_elt_layernorm_out);

    found_subgraph_count++;
  };
  gpd0(graph, handler0);

  GraphPatternDetector gpd1;
  patterns::MultiheadMatmul multihead_matmul(
      gpd1.mutable_pattern(), "remove_padding_recover_padding_pass");
  multihead_matmul();

  std::vector<int64_t> multihead_matmul_input_shape;
  auto handler1 = [&](const GraphPatternDetector::subgraph_t& subgraph,
                      Graph* graph) {
    VLOG(3) << "remove_padding_recover_padding_pass for transformer: "
               "multihead_matmul";

    GET_IR_NODE_FROM_SUBGRAPH(
        multihead_matmul_input, multihead_matmul_input, multihead_matmul);
    GET_IR_NODE_FROM_SUBGRAPH(
        multihead_matmul_op, multihead_matmul_op, multihead_matmul);
    GET_IR_NODE_FROM_SUBGRAPH(
        multihead_matmul_out, multihead_matmul_out, multihead_matmul);

    multihead_matmul_input_shape = multihead_matmul_input->Var()->GetShape();

    insert_remove_padding_op(multihead_matmul_input, multihead_matmul_op);
    insert_recover_padding_op(multihead_matmul_op, multihead_matmul_out);

    found_subgraph_count++;
  };
  gpd1(graph, handler1);

  GraphPatternDetector gpd2;
  patterns::SkipLayernorm skip_layernorm(gpd2.mutable_pattern(),
                                         "remove_padding_recover_padding_pass");
  skip_layernorm();

  auto handler2 = [&](const GraphPatternDetector::subgraph_t& subgraph,
                      Graph* graph) {
    VLOG(3) << "remove_padding_recover_padding_pass for transformer: "
               "skip_layernorm";

    GET_IR_NODE_FROM_SUBGRAPH(
        skip_layernorm_x, skip_layernorm_x, skip_layernorm);
    GET_IR_NODE_FROM_SUBGRAPH(
        skip_layernorm_y, skip_layernorm_y, skip_layernorm);
    GET_IR_NODE_FROM_SUBGRAPH(
        skip_layernorm_op, skip_layernorm_op, skip_layernorm);
    GET_IR_NODE_FROM_SUBGRAPH(
        skip_layernorm_out, skip_layernorm_out, skip_layernorm);

    std::vector<int64_t> skip_layernorm_x_shape =
        skip_layernorm_x->Var()->GetShape();
    check_flag = true;
    if (skip_layernorm_x_shape.size() != multihead_matmul_input_shape.size()) {
      check_flag = false;
      VLOG(3) << "Transformer model remove_padding shape check failed, return "
                 "remove_padding pass.";
      return;
    }
    for (size_t i = 0; i < skip_layernorm_x_shape.size(); ++i) {
      if (skip_layernorm_x_shape[i] != multihead_matmul_input_shape[i]) {
        check_flag = false;
      }
    }
    if (!check_flag) {
      VLOG(3) << "Transformer model remove_padding shape check failed, return "
                 "remove_padding pass.";
      return;
    }
    insert_remove_padding_op(skip_layernorm_x, skip_layernorm_op);
    insert_remove_padding_op(skip_layernorm_y, skip_layernorm_op);
    insert_recover_padding_op(skip_layernorm_op, skip_layernorm_out);
    found_subgraph_count++;
  };
  gpd2(graph, handler2);

  GraphPatternDetector gpd3;
  patterns::Fc fc(gpd3.mutable_pattern(),
                  "remove_padding_recover_padding_pass");
  fc();

  auto handler3 = [&](const GraphPatternDetector::subgraph_t& subgraph,
                      Graph* graph) {
    VLOG(3) << "remove_padding_recover_padding_pass for transformer: fc";

    GET_IR_NODE_FROM_SUBGRAPH(fc_input, fc_input, fc);
    GET_IR_NODE_FROM_SUBGRAPH(fc_op, fc_op, fc);

    std::vector<int64_t> fc_input_shape = fc_input->Var()->GetShape();
    check_flag = true;
    if ((fc_input_shape.size() != multihead_matmul_input_shape.size()) ||
        (fc_input_shape.size() != 3)) {
      check_flag = false;
      VLOG(3) << "Transformer model remove_padding shape check failed, return "
                 "remove_padding pass.";
      return;
    }
    if (fc_input_shape[0] != multihead_matmul_input_shape[0]) {
      check_flag = false;
    }
    if (fc_input_shape[1] != multihead_matmul_input_shape[1]) {
      check_flag = false;
    }
    if ((fc_input_shape[2] != multihead_matmul_input_shape[2]) &&
        (fc_input_shape[2] != 4 * multihead_matmul_input_shape[2])) {
      check_flag = false;
    }

    if (PADDLE_GET_CONST(int, fc_op->Op()->GetAttr("in_num_col_dims")) != 2) {
      check_flag = false;
    }
    if (!check_flag) {
      VLOG(3) << "Transformer model remove_padding shape check failed, return "
                 "remove_padding pass.";
      return;
    }
    insert_remove_padding_op(fc_input, fc_op);
    insert_recover_padding_op(fc_op, fc_op->outputs[0]);
    found_subgraph_count++;
  };
  gpd3(graph, handler3);

  GraphPatternDetector gpd4;
  patterns::Activation activation(gpd4.mutable_pattern(),
                                  "remove_padding_recover_padding_pass");
  activation();

  auto handler4 = [&](const GraphPatternDetector::subgraph_t& subgraph,
                      Graph* graph) {
    VLOG(3)
        << "remove_padding_recover_padding_pass for transformer: activation";

    GET_IR_NODE_FROM_SUBGRAPH(activation_input, activation_input, activation);
    GET_IR_NODE_FROM_SUBGRAPH(activation_op, activation_op, activation);
    GET_IR_NODE_FROM_SUBGRAPH(activation_out, activation_out, activation);

    std::vector<int64_t> activation_input_shape =
        activation_input->Var()->GetShape();
    check_flag = true;
    if ((activation_input_shape.size() !=
         multihead_matmul_input_shape.size()) ||
        (activation_input_shape.size() != 3)) {
      check_flag = false;
      VLOG(3) << "Activation: Transformer model remove_padding "
                 "shape(activation_input_shape.size()) check failed, return "
                 "remove_padding pass.";
      return;
    }
    if (activation_input_shape[0] != multihead_matmul_input_shape[0]) {
      check_flag = false;
    }
    if (activation_input_shape[1] != multihead_matmul_input_shape[1]) {
      check_flag = false;
    }
    if ((activation_input_shape[2] != multihead_matmul_input_shape[2]) &&
        (activation_input_shape[2] != 4 * multihead_matmul_input_shape[2])) {
      check_flag = false;
    }
    if (!check_flag) {
      VLOG(3) << "Activation: Transformer model remove_padding "
                 "shape(activation_input_shape[i]) check failed, return "
                 "remove_padding pass.";
      return;
    }
    insert_remove_padding_op(activation_input, activation_op);
    insert_recover_padding_op(activation_op, activation_out);

    found_subgraph_count++;
  };
  gpd4(graph, handler4);

  GraphPatternDetector gpd5;
  patterns::PrelnEmbEltwiseLayernorm fused_preln_embedding_eltwise_layernorm(
      gpd5.mutable_pattern(), "remove_padding_recover_padding_pass");
  fused_preln_embedding_eltwise_layernorm();

  auto handler5 = [&](const GraphPatternDetector::subgraph_t& subgraph,
                      Graph* graph) {
    VLOG(3) << "remove_padding_recover_padding_pass for transformer: "
               "fused_preln_embedding_eltwise_layernorm";

    GET_IR_NODE_FROM_SUBGRAPH(preln_emb_elt_layernorm_op,
                              preln_emb_elt_layernorm_op,
                              fused_preln_embedding_eltwise_layernorm);
    GET_IR_NODE_FROM_SUBGRAPH(preln_emb_elt_layernorm_out_0,
                              preln_emb_elt_layernorm_out_0,
                              fused_preln_embedding_eltwise_layernorm);
    GET_IR_NODE_FROM_SUBGRAPH(preln_emb_elt_layernorm_out_1,
                              preln_emb_elt_layernorm_out_1,
                              fused_preln_embedding_eltwise_layernorm);

    insert_recover_padding_op(preln_emb_elt_layernorm_op,
                              preln_emb_elt_layernorm_out_0);
    insert_recover_padding_op(preln_emb_elt_layernorm_op,
                              preln_emb_elt_layernorm_out_1);

    found_subgraph_count++;
  };
  gpd5(graph, handler5);

  GraphPatternDetector gpd6;
  patterns::PrelnSkipLayernorm preln_skip_layernorm(
      gpd6.mutable_pattern(), "remove_padding_recover_padding_pass");
  preln_skip_layernorm();

  auto handler6 = [&](const GraphPatternDetector::subgraph_t& subgraph,
                      Graph* graph) {
    VLOG(3) << "remove_padding_recover_padding_pass for transformer: "
               "preln_skip_layernorm";

    GET_IR_NODE_FROM_SUBGRAPH(
        preln_skip_layernorm_x, preln_skip_layernorm_x, preln_skip_layernorm);
    GET_IR_NODE_FROM_SUBGRAPH(
        preln_skip_layernorm_y, preln_skip_layernorm_y, preln_skip_layernorm);
    GET_IR_NODE_FROM_SUBGRAPH(
        preln_skip_layernorm_op, preln_skip_layernorm_op, preln_skip_layernorm);
    GET_IR_NODE_FROM_SUBGRAPH(preln_skip_layernorm_out_0,
                              preln_skip_layernorm_out_0,
                              preln_skip_layernorm);
    GET_IR_NODE_FROM_SUBGRAPH(preln_skip_layernorm_out_1,
                              preln_skip_layernorm_out_1,
                              preln_skip_layernorm);

    std::vector<int64_t> skip_layernorm_x_shape =
        preln_skip_layernorm_x->Var()->GetShape();
    check_flag = true;
    if (skip_layernorm_x_shape.size() != multihead_matmul_input_shape.size()) {
      check_flag = false;
      VLOG(3) << "Transformer model remove_padding shape check failed, return "
                 "remove_padding pass.";
      return;
    }
    for (size_t i = 0; i < skip_layernorm_x_shape.size(); ++i) {
      if (skip_layernorm_x_shape[i] != multihead_matmul_input_shape[i]) {
        check_flag = false;
      }
    }
    if (!check_flag) {
      VLOG(3) << "Transformer model remove_padding shape check failed, return "
                 "remove_padding pass.";
      return;
    }
    insert_remove_padding_op(preln_skip_layernorm_x, preln_skip_layernorm_op);
    insert_remove_padding_op(preln_skip_layernorm_y, preln_skip_layernorm_op);
    insert_recover_padding_op(preln_skip_layernorm_op,
                              preln_skip_layernorm_out_0);
    insert_recover_padding_op(preln_skip_layernorm_op,
                              preln_skip_layernorm_out_1);
    found_subgraph_count++;
  };
  gpd6(graph, handler6);

  GraphPatternDetector gpd7;
  patterns::FusedTokenPrune fused_token_prune(
      gpd7.mutable_pattern(), "remove_padding_recover_padding_pass");
  fused_token_prune();

  auto handler7 = [&](const GraphPatternDetector::subgraph_t& subgraph,
                      Graph* graph) {
    VLOG(3) << "remove_padding_recover_padding_pass for transformer: "
               "fused_token_prune";

    GET_IR_NODE_FROM_SUBGRAPH(
        fused_token_prune_input, fused_token_prune_input, fused_token_prune);
    GET_IR_NODE_FROM_SUBGRAPH(
        fused_token_prune_op, fused_token_prune_op, fused_token_prune);
    GET_IR_NODE_FROM_SUBGRAPH(
        fused_token_prune_output, fused_token_prune_output, fused_token_prune);

    std::vector<int64_t> fused_token_prune_input_shape =
        fused_token_prune_input->Var()->GetShape();
    check_flag = true;
    if (fused_token_prune_input_shape.size() !=
        multihead_matmul_input_shape.size()) {
      check_flag = false;
      VLOG(3) << "Transformer model remove_padding shape check failed, return "
                 "remove_padding pass.";
      return;
    }
    for (size_t i = 0; i < fused_token_prune_input_shape.size(); ++i) {
      if (fused_token_prune_input_shape[i] != multihead_matmul_input_shape[i]) {
        check_flag = false;
      }
    }
    if (!check_flag) {
      VLOG(3) << "Transformer model remove_padding shape check failed, return "
                 "remove_padding pass.";
      return;
    }
    insert_recover_padding_op(fused_token_prune_op, fused_token_prune_output);
    found_subgraph_count++;
  };
  gpd7(graph, handler7);

  AddStatis(found_subgraph_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(remove_padding_recover_padding_pass,
              paddle::framework::ir::RemovePaddingRecoverPaddingPass);
