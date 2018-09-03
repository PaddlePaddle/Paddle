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

#include "paddle/fluid/framework/ir/fc_lstm_fuse_pass.h"
#include <string>
#include "paddle/fluid/framework/lod_tensor.h"

namespace paddle {
namespace framework {
namespace ir {

std::string GenNodeName(const std::string& prefix, const std::string& name) {
  return prefix + "/" + name;
}

void BuildPattern(PDPattern* pattern, const std::string& name_scope,
                  bool with_fc_bias) {
  PDNode* x = pattern->NewNode(name_scope, "x")
                  ->assert_is_op_input("mul")
                  ->assert_var_not_persistable();
  auto* fc_out = patterns::FC(pattern, name_scope, x, with_fc_bias);
  fc_out->AsIntermediate();  // fc_out is a tmp var, will be removed after fuse.
  patterns::LSTM(pattern, name_scope, fc_out);
  // LOG(INFO) << "\n" << pattern->DotString();
}

int BuildFusion(Graph* graph, const std::string& name_scope, Scope* scope,
                bool with_fc_bias) {
  GraphPatternDetector gpd;
  auto* pattern = gpd.mutable_pattern();

  BuildPattern(pattern, name_scope, with_fc_bias);

  // Create New OpDesc
  auto lstm_creator = [&](int lstm, int input, int weight_x, int weight_h,
                          int bias, int hidden, int cell, int xx, int fc_bias) {
#define GET_NODE(x) auto* x##_n = graph->RetriveNode(x);
    GET_NODE(input);
    GET_NODE(weight_x);
    GET_NODE(weight_h);
    GET_NODE(bias);
    GET_NODE(hidden);
    GET_NODE(cell);
    GET_NODE(xx);
    GET_NODE(lstm);

    OpDesc op_desc;
    op_desc.SetType("fusion_lstm");
#define SET_IN(Key, node__) op_desc.SetInput(#Key, {node__##_n->Name()});
    SET_IN(X, input);
    SET_IN(WeightX, weight_x);
    SET_IN(WeightH, weight_h);
    SET_IN(Bias, bias);
#undef SET_IN
    if (with_fc_bias) {
      // Add FC-bias with LSTM-bias and create a new weight
      PADDLE_ENFORCE(scope);
      const std::string& new_bias_var = name_scope + "_bias.new";
      auto* bias_var = scope->Var(new_bias_var);
      PADDLE_ENFORCE(bias_var);
      auto* bias_tensor = bias_var->GetMutable<framework::LoDTensor>();
      auto* lstm_bias_var = scope->FindVar(bias_n->Name());
      PADDLE_ENFORCE(lstm_bias_var);
      const auto& lstm_bias_tensor = lstm_bias_var->Get<framework::LoDTensor>();
      bias_tensor->Resize(lstm_bias_tensor.dims());

      GET_NODE(fc_bias);
      auto* fc_bias_var = scope->FindVar(fc_bias_n->Name());
      const auto& fc_bias_tensor = fc_bias_var->Get<framework::LoDTensor>();

      auto* data = bias_tensor->mutable_data<float>(platform::CPUPlace());

      for (int i = 0; i < bias_tensor->numel(); i++) {
        data[i] =
            fc_bias_tensor.data<float>()[i] + lstm_bias_tensor.data<float>()[i];
      }
      op_desc.SetInput("Bias", {new_bias_var});
    }

#undef GET_NODE

    op_desc.SetInput("H0", {});
    op_desc.SetInput("C0", {});
    op_desc.SetOutput("Hidden", {hidden_n->Name()});
    op_desc.SetOutput("Cell", {cell_n->Name()});
    op_desc.SetOutput("XX", {xx_n->Name()});
    op_desc.SetOutput("BatchedInput", {"blstm_0.tmp_2"});
    op_desc.SetAttr("is_reverse", lstm_n->Op()->GetAttr("is_reverse"));
    op_desc.SetAttr("use_peepholes", lstm_n->Op()->GetAttr("use_peepholes"));
    // TODO(TJ): get from attr
    op_desc.SetAttr("use_seq", true);

#define TMP_NAME(x) "at.new.tmp." #x
#define OP_SET_OUT(x) op_desc.SetOutput(#x, {TMP_NAME(x)})
    OP_SET_OUT(BatchedCell);
    OP_SET_OUT(BatchedHidden);
    OP_SET_OUT(ReorderedH0);
    OP_SET_OUT(ReorderedC0);
#undef OP_SET_OUT

    auto* op = graph->CreateOpNode(&op_desc);
    PADDLE_ENFORCE(graph->Has(kParamScopeAttr));
    auto* scope = graph->Get<Scope*>(kParamScopeAttr);

#define TMP_NEW(x) scope->Var(TMP_NAME(x))->GetMutable<LoDTensor>()
    TMP_NEW(BatchedCell);
    TMP_NEW(BatchedHidden);
    TMP_NEW(ReorderedH0);
    TMP_NEW(ReorderedC0);
#undef TMP_NEW
#undef TMP_NAME

#define LINK_TO(a, b)      \
  a->outputs.push_back(b); \
  b->inputs.push_back(a);
    LINK_TO(input_n, op);
    LINK_TO(weight_x_n, op);
    LINK_TO(weight_h_n, op);
    LINK_TO(bias_n, op);
    LINK_TO(op, hidden_n);
#undef LINK_TO
    return op;
  };

  int fusion_count{0};

  auto fc_no_bias_handler = [&](
      const GraphPatternDetector::subgraph_t& subgraph, Graph* g) {
#define GET_NODE(name__)                                \
  std::string name__##key = name_scope + "/" + #name__; \
  auto* name__##n = pattern->RetrieveNode(name__##key); \
  PADDLE_ENFORCE(name__##n);                            \
  PADDLE_ENFORCE(subgraph.count(name__##n));            \
  Node* name__##_n = subgraph.at(name__##n);            \
  int name__ __attribute__((unused)) = name__##_n->id();

    GET_NODE(x);
    GET_NODE(w);
    GET_NODE(mul);
    GET_NODE(fc_out);
    GET_NODE(Weight);
    GET_NODE(lstm);
    GET_NODE(Bias);
    GET_NODE(Hidden);
    GET_NODE(Cell);

    if (with_fc_bias) {
      GET_NODE(fc_bias);
      lstm_creator(lstm, x, w, Weight, Bias, Hidden, Cell, fc_out, fc_bias);
    } else {
      lstm_creator(lstm, x, w, Weight, Bias, Hidden, Cell, fc_out, -1);
    }
#undef GET_NODE

    // Remove unneeded nodes.
    std::unordered_set<const Node*> marked_nodes({mul_n, lstm_n});

    GraphSafeRemoveNodes(graph, marked_nodes);

    ++fusion_count;
  };

  gpd(graph, fc_no_bias_handler);

  return fusion_count;
}

std::unique_ptr<ir::Graph> MulLstmFusePass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  FusePassBase::Init(name_scope_, graph.get());

  int fusion_count = BuildFusion(graph.get(), name_scope_, param_scope(),
                                 false /*with_fc_bias*/);

  AddStatis(fusion_count);
  return graph;
}

std::unique_ptr<ir::Graph> FCLstmFusePass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  FusePassBase::Init(name_scope_, graph.get());

  int fusion_count = BuildFusion(graph.get(), name_scope_, param_scope(),
                                 true /*with_fc_bias*/);

  AddStatis(fusion_count);
  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(mul_lstm_fuse_pass, paddle::framework::ir::MulLstmFusePass);
REGISTER_PASS(fc_lstm_fuse_pass, paddle::framework::ir::FCLstmFusePass);
