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

#include "paddle/fluid/framework/ir/fc_gru_fuse_pass.h"
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
  patterns::GRU(pattern, name_scope, fc_out);
  VLOG(3) << "\n" << pattern->DotString();
}

int BuildFusion(Graph* graph, const std::string& name_scope, Scope* scope,
                bool with_fc_bias) {
  GraphPatternDetector gpd;
  auto* pattern = gpd.mutable_pattern();

  BuildPattern(pattern, name_scope, with_fc_bias);

  // Create New OpDesc
  auto gru_creater = [&](int gru, int x, int weight_x, int weight_h, int bias,
                         int hidden, int fc_bias) {
#define GET_NODE(x) auto* x##_n = graph->RetriveNode(x);
    GET_NODE(x);
    GET_NODE(weight_x);
    GET_NODE(weight_h);
    GET_NODE(bias);
    GET_NODE(hidden);
    GET_NODE(gru);

    OpDesc op_desc;
    op_desc.SetType("fusion_gru");
#define SET_IN(Key, node__) op_desc.SetInput(#Key, {node__##_n->Name()});
    SET_IN(X, x);
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
      auto* gru_bias_var = scope->FindVar(bias_n->Name());
      PADDLE_ENFORCE(gru_bias_var);
      const auto& gru_bias_tenosr = gru_bias_var->Get<framework::LoDTensor>();
      bias_tensor->Resize(gru_bias_tenosr.dims());

      GET_NODE(fc_bias);
      auto* fc_bias_var = scope->FindVar(fc_bias_n->Name());
      const auto& fc_bias_tensor = fc_bias_var->Get<framework::LoDTensor>();
      // new bias = fc bias + gru bias
      auto* data = bias_tensor->mutable_data<float>(platform::CPUPlace());
      for (int i = 0; i < bias_tensor->numel(); i++) {
        data[i] =
            fc_bias_tensor.data<float>()[i] + gru_bias_tenosr.data<float>()[i];
      }
      op_desc.SetInput("Bias", {new_bias_var});
    }
#undef GET_NODE

    op_desc.SetInput("H0", {});
    op_desc.SetOutput("Hidden", {hidden_n->Name()});
    op_desc.SetAttr("is_reverse", gru_n->Op()->GetAttr("is_reverse"));
    // TODO(TJ): This should be a option for infer
    op_desc.SetAttr("use_seq", true);

    // Create temp variables.
    // TODO(TJ): clean code
    scope->Var(name_scope + "/ReorderedH0.new")
        ->GetMutable<framework::LoDTensor>();
    scope->Var(name_scope + "/XX.new")->GetMutable<framework::LoDTensor>();
    scope->Var(name_scope + "/BatchedInput.new")
        ->GetMutable<framework::LoDTensor>();
    scope->Var(name_scope + "/BatchedOut.new")
        ->GetMutable<framework::LoDTensor>();
    op_desc.SetOutput("ReorderedH0", {name_scope + "/ReorderedH0.new"});
    op_desc.SetOutput("XX", {name_scope + "/XX.new"});
    op_desc.SetOutput("BatchedInput", {name_scope + "/BatchedInput.new"});
    op_desc.SetOutput("BatchedOut", {name_scope + "/BatchedOut.new"});

    auto* op = graph->CreateOpNode(&op_desc);
    PADDLE_ENFORCE(graph->Has(kParamScopeAttr));
    auto* scope = graph->Get<Scope*>(kParamScopeAttr);

    IR_NODE_LINK_TO(x_n, op);
    IR_NODE_LINK_TO(weight_x_n, op);
    IR_NODE_LINK_TO(weight_h_n, op);
    IR_NODE_LINK_TO(bias_n, op);
    IR_NODE_LINK_TO(op, hidden_n);
    // h0?
    return op;
  };

  int fusion_count{0};
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
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
    GET_NODE(gru);
    GET_NODE(Bias);
    GET_NODE(Hidden);

    if (with_fc_bias) {
      GET_NODE(fc_bias);
      GET_NODE(elementwise_add);
      gru_creater(gru, x, w, Weight, Bias, Hidden, fc_bias);
      // Remove unneeded nodes.
      std::unordered_set<const Node*> marked_nodes(
          {mul_n, gru_n, elementwise_add_n});
      GraphSafeRemoveNodes(graph, marked_nodes);
    } else {
      gru_creater(gru, x, w, Weight, Bias, Hidden, -1);
      // Remove unneeded nodes.
      std::unordered_set<const Node*> marked_nodes({mul_n, gru_n});
      GraphSafeRemoveNodes(graph, marked_nodes);
    }
#undef GET_NODE

    ++fusion_count;
  };

  gpd(graph, handler);

  return fusion_count;
}

std::unique_ptr<ir::Graph> MulGRUFusePass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  FusePassBase::Init(name_scope_, graph.get());

  int fusion_count = BuildFusion(graph.get(), name_scope_, param_scope(),
                                 false /*with_fc_bias*/);

  AddStatis(fusion_count);
  return graph;
}

std::unique_ptr<ir::Graph> FCGRUFusePass::ApplyImpl(
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

REGISTER_PASS(mul_lstm_fuse_pass, paddle::framework::ir::MulGRUFusePass);
REGISTER_PASS(fc_lstm_fuse_pass, paddle::framework::ir::FCGRUFusePass);
