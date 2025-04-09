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

#include <string>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

namespace paddle {
namespace framework {
namespace ir {

static int BuildFusion(Graph* graph,
                       const std::string& name_scope,
                       Scope* scope,
                       bool with_fc_bias) {
  GraphPatternDetector gpd;
  auto* pattern = gpd.mutable_pattern();

  // Build pattern
  PDNode* x = pattern->NewNode(patterns::PDNodeName(name_scope, "x"))
                  ->assert_is_op_input("lookup_table_v2")
                  ->assert_var_not_persistable();
  patterns::Embedding embedding_pattern(pattern, name_scope);
  // Intermediate can only be for val that are not used anywhere but
  // lookup table output may go into other LSTM (for reverse direction)
  auto* embedding_out = embedding_pattern(x);
  patterns::FC fc_pattern(pattern, name_scope);

  // fc_out is a tmp var, will be removed after fuse, so marked as intermediate.
  auto* fc_out = fc_pattern(embedding_out, with_fc_bias, /* with_relu */ false)
                     ->AsIntermediate();
  patterns::LSTM lstm_pattern(pattern, name_scope);
  lstm_pattern(fc_out);

  // Create New OpDesc
  auto embedding_lstm_creator = [&](Node* embedding,
                                    Node* W,
                                    Node* lstm,
                                    Node* input,
                                    Node* weight_x,
                                    Node* weight_h,
                                    Node* bias,
                                    Node* hidden,
                                    Node* cell,
                                    Node* xx,
                                    Node* fc_bias) {
    OpDesc op_desc;
    op_desc.SetType("fused_embedding_fc_lstm");
#define SET_IN(Key, node__) op_desc.SetInput(#Key, {node__->Name()});
    SET_IN(Ids, input);
    SET_IN(WeightH, weight_h);
    // Need to have this passed as We need Wc data for peephole connections
    SET_IN(Bias, bias);
#undef SET_IN

    // Multiply embeddings with Weights
    PADDLE_ENFORCE_NOT_NULL(
        scope, common::errors::InvalidArgument("Scope cannot be nullptr."));
    const std::string& embeddings = patterns::UniqueKey("Embeddings");
    auto* embeddings_var = scope->Var(embeddings);
    PADDLE_ENFORCE_NOT_NULL(
        embeddings_var,
        common::errors::InvalidArgument(
            "Embeddings variable's pointer cannot be nullptr."));
    auto* embeddings_tensor = embeddings_var->GetMutable<phi::DenseTensor>();
    // Get WeightX size: [single_embedding, fc_size]
    // and embedding size: [dict_size, single_embedding]
    // and create new size of embeddings eg. [dict_size , hidden_size]
    auto* embedding_var = scope->FindVar(W->Name());
    PADDLE_ENFORCE_NOT_NULL(
        embedding_var,
        common::errors::InvalidArgument(
            "Embedding variable's pointer cannot be nullptr."));
    const auto& embedding_tensor = embedding_var->Get<phi::DenseTensor>();

    const auto& weightx_tensor =
        scope->FindVar(weight_x->Name())->Get<phi::DenseTensor>();
    embeddings_tensor->Resize(
        {embedding_tensor.dims()[0], weightx_tensor.dims()[1]});

    // Multiply embeddings via WeightsX and add bias
    auto embedding_data = embedding_tensor.data<float>();
    auto weightx_data = weightx_tensor.data<float>();
    auto embeddings_data =
        embeddings_tensor->mutable_data<float>(phi::CPUPlace());

    // Adding biases to GEMM result to be
    auto* lstm_bias_var = scope->FindVar(bias->Name());
    PADDLE_ENFORCE_NOT_NULL(lstm_bias_var,
                            common::errors::InvalidArgument(
                                "Lstm bias var ptr cannot be nullptr."));
    const auto& lstm_bias_tensor = lstm_bias_var->Get<phi::DenseTensor>();

    auto alpha = 1.0f;
    auto beta = 1.0f;
    int m = static_cast<int>(embedding_tensor.dims()[0]);
    int n = static_cast<int>(weightx_tensor.dims()[1]);
    int k = static_cast<int>(embedding_tensor.dims()[1]);

    // Copy only gate biases values (only actual bias data, not peephole
    // weights)
    std::vector<float> combined_biases;
    combined_biases.reserve(n);
    std::copy_n(
        lstm_bias_tensor.data<float>(), n, std::back_inserter(combined_biases));

    if (with_fc_bias) {
      // Add FC-bias with LSTM-bias (into GEMM result to be)
      auto* fc_bias_var = scope->FindVar(fc_bias->Name());  // NOLINT
      const auto& fc_bias_tensor = fc_bias_var->Get<phi::DenseTensor>();
      for (int i = 0; i < fc_bias_tensor.numel(); i++) {
        combined_biases[i] += fc_bias_tensor.data<float>()[i];
      }
    }

    // broadcast biases
    std::vector<float> ones(m, 1.0f);
    phi::funcs::CBlas<float>::GEMM(CblasRowMajor,
                                   CblasNoTrans,
                                   CblasNoTrans,
                                   m,
                                   n,
                                   1,
                                   alpha,
                                   &ones[0],
                                   1,
                                   &combined_biases[0],
                                   n,
                                   0.0f,
                                   embeddings_data,
                                   n);

    // Wx*embeddings + biases
    phi::funcs::CBlas<float>::GEMM(CblasRowMajor,
                                   CblasNoTrans,
                                   CblasNoTrans,
                                   m,
                                   n,
                                   k,
                                   alpha,
                                   embedding_data,
                                   k,
                                   weightx_data,
                                   n,
                                   beta,
                                   embeddings_data,
                                   n);
    op_desc.SetInput("Embeddings", {embeddings});

    op_desc.SetInput("H0", {});
    op_desc.SetInput("C0", {});
    op_desc.SetOutput("Hidden", {hidden->Name()});
    op_desc.SetOutput("Cell", {cell->Name()});
    op_desc.SetOutput("XX", {xx->Name()});
    op_desc.SetAttr("is_reverse", lstm->Op()->GetAttr("is_reverse"));
    op_desc.SetAttr("use_peepholes", lstm->Op()->GetAttr("use_peepholes"));
    // TODO(TJ): get from attr
    op_desc.SetAttr("use_seq", true);

// Create temp variables.
#define OP_SET_OUT(x)                            \
  const std::string x = patterns::UniqueKey(#x); \
  op_desc.SetOutput(#x, {x});

    OP_SET_OUT(BatchedGate);
    OP_SET_OUT(BatchCellPreAct);
    OP_SET_OUT(BatchedInput);
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

#define IR_NODE(x)                                 \
  VarDesc key_##x(x);                              \
  key_##x.SetPersistable(false);                   \
  auto* node_##x = graph->CreateVarNode(&key_##x); \
  IR_NODE_LINK_TO(op, node_##x);

    IR_NODE(BatchedGate);
    IR_NODE(BatchCellPreAct);
    IR_NODE(BatchedInput);
    IR_NODE(BatchedCell);
    IR_NODE(BatchedHidden);
    IR_NODE(ReorderedH0);
    IR_NODE(ReorderedC0);
#undef IR_NODE

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

    auto is_sparse =
        PADDLE_GET_CONST(bool, lookup_table->Op()->GetAttr("is_sparse"));
    auto is_distributed =
        PADDLE_GET_CONST(bool, lookup_table->Op()->GetAttr("is_distributed"));

    if (is_sparse || is_distributed) {
      VLOG(4) << "Only dense embedding is supported in oneDNN";
      return;
    }

    if (with_fc_bias) {
      GET_IR_NODE_FROM_SUBGRAPH(fc_out, elementwise_add_out, fc_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(fc_bias, bias, fc_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(elementwise_add, elementwise_add, fc_pattern);
      embedding_lstm_creator(lookup_table,
                             W,
                             lstm,
                             subgraph.at(x),
                             w,
                             Weight,
                             Bias,
                             Hidden,
                             Cell,
                             fc_out,
                             fc_bias);
      std::unordered_set<const Node*> marked_nodes(
          {mul, lstm, elementwise_add, fc_bias});
      GraphSafeRemoveNodes(graph, marked_nodes);
    } else {
      GET_IR_NODE_FROM_SUBGRAPH(fc_out, mul_out, fc_pattern);
      embedding_lstm_creator(lookup_table,
                             W,
                             lstm,
                             subgraph.at(x),
                             w,
                             Weight,
                             Bias,
                             Hidden,
                             Cell,
                             fc_out,
                             nullptr);
      std::unordered_set<const Node*> marked_nodes({mul, lstm});
      GraphSafeRemoveNodes(graph, marked_nodes);
    }

    ++fusion_count;
  };

  gpd(graph, handler);

  return fusion_count;
}

void EmbeddingFCLSTMFusePass::ApplyImpl(ir::Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);

  int fusion_count =
      BuildFusion(graph, name_scope_, param_scope(), true /*with_fc_bias*/);

  AddStatis(fusion_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(embedding_fc_lstm_fuse_pass,
              paddle::framework::ir::EmbeddingFCLSTMFusePass);
REGISTER_PASS_CAPABILITY(embedding_fc_lstm_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("lookup_table_v2", 0)
            .EQ("mul", 0)
            .LE("elementwise_add", 1)
            .EQ("lstm", 0)
            .EQ("fused_embedding_fc_lstm", 0));
