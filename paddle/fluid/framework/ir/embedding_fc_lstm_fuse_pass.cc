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

#include "paddle/fluid/framework/ir/embedding_fc_lstm_fuse_pass.h"
#include <algorithm>
#include <string>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/lod_tensor.h"

#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/cpu_vec.h"
#include "paddle/fluid/operators/math/fc_compute.h"
#include "paddle/fluid/platform/cpu_info.h"

namespace paddle {
namespace framework {
namespace ir {

static int BuildFusion(Graph* graph, const std::string& name_scope,
                       Scope* scope, bool with_fc_bias) {
  GraphPatternDetector gpd;
  auto* pattern = gpd.mutable_pattern();

  // Build pattern
  PDNode* x = pattern->NewNode(patterns::PDNodeName(name_scope, "x"))
                  ->assert_is_op_input("lookup_table")
                  ->assert_var_not_persistable();
  patterns::Embedding embedding_pattern(pattern, name_scope);
  // TODO(jczaja): Intermediate can only be for val that are not used anywhere
  //               but lookup table output may go into other LSTM (for reverse
  //               direction)
  auto* embedding_out = embedding_pattern(x);
  patterns::FC fc_pattern(pattern, name_scope);

  // fc_out is a tmp var, will be removed after fuse, so marked as intermediate.
  auto* fc_out = fc_pattern(embedding_out, with_fc_bias)->AsIntermediate();
  patterns::LSTM lstm_pattern(pattern, name_scope);
  lstm_pattern(fc_out);

  // Create New OpDesc
  auto embedding_lstm_creator = [&](Node* embedding, Node* W, Node* lstm,
                                    Node* input, Node* weight_x, Node* weight_h,
                                    Node* bias, Node* hidden, Node* cell,
                                    Node* xx, Node* fc_bias) {
    OpDesc op_desc;
    op_desc.SetType("fused_embedding_fc_lstm");
#define SET_IN(Key, node__) op_desc.SetInput(#Key, {node__->Name()});
    SET_IN(Ids, input);
    SET_IN(WeightH, weight_h);
    // Neet to have this passed as We need Wc data for peephole connections
    SET_IN(Bias, bias);
#undef SET_IN

    // Multiply embeddings with Weights
    PADDLE_ENFORCE(scope);
    const std::string& embeddings = patterns::UniqueKey("Embeddings");
    auto* embeddings_var = scope->Var(embeddings);
    PADDLE_ENFORCE(embeddings_var);
    auto* embeddings_tensor =
        embeddings_var->GetMutable<framework::LoDTensor>();
    // Get WeightX size: [single_embedding, fc_size]
    // and embedding size: [dict_size, single_embedding]
    // and create new size of embeddings eg. [dict_size , hidden_size]
    auto* embedding_var = scope->FindVar(W->Name());
    PADDLE_ENFORCE(embedding_var);
    const auto& embedding_tensor = embedding_var->Get<framework::LoDTensor>();

    const auto& weightx_tensor =
        scope->FindVar(weight_x->Name())->Get<framework::LoDTensor>();
    embeddings_tensor->Resize(
        {embedding_tensor.dims()[0], weightx_tensor.dims()[1]});

    // Multiplie embeddings via WeightsX and add bias
    auto embedding_data = embedding_tensor.data<float>();
    auto weightx_data = weightx_tensor.data<float>();
    auto embeddings_data =
        embeddings_tensor->mutable_data<float>(platform::CPUPlace());

    // Adding biases to GEMM result to be
    auto* lstm_bias_var = scope->FindVar(bias->Name());
    PADDLE_ENFORCE(lstm_bias_var);
    const auto& lstm_bias_tensor = lstm_bias_var->Get<framework::LoDTensor>();

    auto alpha = 1.0f;
    auto beta = 1.0f;
    int m = embedding_tensor.dims()[0];
    int n = weightx_tensor.dims()[1];
    int k = embedding_tensor.dims()[1];

    // Copy only gate biases values (only actual bias data, not peephole
    // weights)
    std::vector<float> combined_biases;
    combined_biases.reserve(n);
    std::copy_n(lstm_bias_tensor.data<float>(), n,
                std::back_inserter(combined_biases));

    if (with_fc_bias) {
      // Add FC-bias with LSTM-bias (into GEMM result to be)
      auto* fc_bias_var = scope->FindVar(fc_bias->Name());
      const auto& fc_bias_tensor = fc_bias_var->Get<framework::LoDTensor>();
      for (int i = 0; i < fc_bias_tensor.numel(); i++) {
        combined_biases[i] += fc_bias_tensor.data<float>()[i];
      }
    }

    // broadcast biases
    std::vector<float> ones(m, 1.0f);
    paddle::operators::math::CBlas<float>::GEMM(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, 1, alpha, &ones[0], 1,
        &combined_biases[0], n, 0.0f, embeddings_data, n);

    // Wx*embeddings + biases
    paddle::operators::math::CBlas<float>::GEMM(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha,
        embedding_data, k, weightx_data, n, beta, embeddings_data, n);
    op_desc.SetInput("Embeddings", {embeddings});

    // Create temp variables.
    const std::string BatchedInput = patterns::UniqueKey("BatchedInput");
    const std::string BatchedCellPreAct =
        patterns::UniqueKey("BatchedCellPreAct");
    const std::string BatchedGate = patterns::UniqueKey("BatchedGate");

    scope->Var(BatchedInput)->GetMutable<framework::LoDTensor>();
    scope->Var(BatchedCellPreAct)->GetMutable<framework::LoDTensor>();
    scope->Var(BatchedGate)->GetMutable<framework::LoDTensor>();

    op_desc.SetInput("H0", {});
    op_desc.SetInput("C0", {});
    op_desc.SetOutput("Hidden", {hidden->Name()});
    op_desc.SetOutput("Cell", {cell->Name()});
    op_desc.SetOutput("XX", {xx->Name()});
    op_desc.SetOutput("BatchedGate", {BatchedGate});
    op_desc.SetOutput("BatchCellPreAct", {BatchedCellPreAct});
    op_desc.SetOutput("BatchedInput", {BatchedInput});
    op_desc.SetAttr("is_reverse", lstm->Op()->GetAttr("is_reverse"));
    op_desc.SetAttr("use_peepholes", lstm->Op()->GetAttr("use_peepholes"));
    // TODO(TJ): get from attr
    op_desc.SetAttr("use_seq", true);

    PADDLE_ENFORCE(graph->Has(kParamScopeAttr));
    auto* scope = graph->Get<Scope*>(kParamScopeAttr);
#define OP_SET_OUT(x)                            \
  const std::string x = patterns::UniqueKey(#x); \
  op_desc.SetOutput(#x, {x});                    \
  scope->Var(x)->GetMutable<LoDTensor>()
    OP_SET_OUT(BatchedCell);
    OP_SET_OUT(BatchedHidden);
    OP_SET_OUT(ReorderedH0);
    OP_SET_OUT(ReorderedC0);
#undef OP_SET_OUT

    auto* op = graph->CreateOpNode(&op_desc);
    IR_NODE_LINK_TO(input, op);
    IR_NODE_LINK_TO(weight_x, op);
    IR_NODE_LINK_TO(weight_h, op);
    IR_NODE_LINK_TO(bias, op);
    IR_NODE_LINK_TO(op, hidden);
    return op;
  };

  int fusion_count{0};

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(lstm, lstm, lstm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(Weight, Weight, lstm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(Bias, Bias, lstm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(Cell, Cell, lstm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(Hidden, Hidden, lstm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(lookup_table, lookup_table, embedding_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(W, W, embedding_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(w, w, fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul, mul, fc_pattern);

    // TODO(jczaja): Add support for is_sparse / is_distributed
    auto is_sparse = boost::get<bool>(lookup_table->Op()->GetAttr("is_sparse"));
    auto is_distributed =
        boost::get<bool>(lookup_table->Op()->GetAttr("is_distributed"));

    if (is_sparse == true || is_distributed == true) {
      return;
    }

    if (with_fc_bias) {
      GET_IR_NODE_FROM_SUBGRAPH(fc_out, Out, fc_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(fc_bias, bias, fc_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(elementwise_add, elementwise_add, fc_pattern);
      embedding_lstm_creator(lookup_table, W, lstm, subgraph.at(x), w, Weight,
                             Bias, Hidden, Cell, fc_out, fc_bias);
      // Remove unneeded nodes.
      // TODO(jczaja): Proper removing of lookup table
      std::unordered_set<const Node*> marked_nodes(
          // {lookup_table, mul, lstm, elementwise_add, fc_bias, W});
          {mul, lstm, elementwise_add, fc_bias});
      GraphSafeRemoveNodes(graph, marked_nodes);
    } else {
      GET_IR_NODE_FROM_SUBGRAPH(fc_out, mul_out, fc_pattern);
      embedding_lstm_creator(lookup_table, W, lstm, subgraph.at(x), w, Weight,
                             Bias, Hidden, Cell, fc_out, nullptr);
      // Remove unneeded nodes.
      // TODO(jczaja): Proper removing of lookup table
      // std::unordered_set<const Node*> marked_nodes({lookup_table, W, mul,
      // lstm});
      std::unordered_set<const Node*> marked_nodes({mul, lstm});
      GraphSafeRemoveNodes(graph, marked_nodes);
    }

    ++fusion_count;
  };

  gpd(graph, handler);

  return fusion_count;
}

ir::Graph* EmbeddingFCLSTMFusePass::ApplyImpl(ir::Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);

  int fusion_count =
      BuildFusion(graph, name_scope_, param_scope(), true /*with_fc_bias*/);

  AddStatis(fusion_count);
  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(embedding_fc_lstm_fuse_pass,
              paddle::framework::ir::EmbeddingFCLSTMFusePass);
