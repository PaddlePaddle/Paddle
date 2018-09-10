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

static void BuildPattern(PDPattern* pattern, const std::string& name_scope,
                         bool with_fc_bias) {
  PDNode* x = pattern->NewNode(name_scope, "x")
                  ->assert_is_op_input("mul")
                  ->assert_var_not_persistable();
  auto* fc_out = patterns::FC(pattern, name_scope, x, with_fc_bias);
  fc_out->AsIntermediate();  // fc_out is a tmp var, will be removed after fuse.
  patterns::GRU(pattern, name_scope, fc_out);
  VLOG(3) << "fc_gru pattern \n" << pattern->DotString();
}

static int BuildFusion(Graph* graph, const std::string& name_scope,
                       Scope* scope, bool with_fc_bias) {
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

#define NEW_NAME(x) name_scope + "/at." #x ".new"
#define SET_IN(Key, node__) op_desc.SetInput(#Key, {node__##_n->Name()});
    SET_IN(X, x);
    SET_IN(WeightX, weight_x);
    SET_IN(WeightH, weight_h);
    if (with_fc_bias) {
      op_desc.SetInput("Bias", {NEW_NAME(bias) + bias_n->Name()});
    } else {
      SET_IN(Bias, bias);
    }
#undef SET_IN
    op_desc.SetInput("H0", {});
    op_desc.SetOutput("Hidden", {hidden_n->Name()});
    op_desc.SetAttr("is_reverse", gru_n->Op()->GetAttr("is_reverse"));
    // TODO(TJ): This should be a option for infer
    op_desc.SetAttr("use_seq", true);

#define SET_IMTERMEDIATE_OUT(key) op_desc.SetOutput(#key, {NEW_NAME(key)})
    SET_IMTERMEDIATE_OUT(ReorderedH0);
    SET_IMTERMEDIATE_OUT(XX);
    SET_IMTERMEDIATE_OUT(BatchedInput);
    SET_IMTERMEDIATE_OUT(BatchedOut);
#undef SET_IMTERMEDIATE_OUT

    auto* op = graph->CreateOpNode(&op_desc);
    PADDLE_ENFORCE(graph->Has(kParamScopeAttr));
    auto* scope = graph->Get<Scope*>(kParamScopeAttr);
    PADDLE_ENFORCE(scope);
    if (with_fc_bias) {
      // Fusion GRU bias = fcbias + grubias
      auto* fusion_bias_var = scope->Var(NEW_NAME(bias) + bias_n->Name());
      auto* out_bias_tensor =
          fusion_bias_var->GetMutable<framework::LoDTensor>();
      PADDLE_ENFORCE(fusion_bias_var);
      GET_NODE(fc_bias);
      PADDLE_ENFORCE(fc_bias_n);
      auto* gru_bias_var = scope->FindVar(bias_n->Name());
      auto* fc_bias_var = scope->FindVar(fc_bias_n->Name());
      PADDLE_ENFORCE(gru_bias_var);
      PADDLE_ENFORCE(fc_bias_var);
      const auto& gru_bias_tenosr = gru_bias_var->Get<framework::LoDTensor>();
      const auto& fc_bias_tensor = fc_bias_var->Get<framework::LoDTensor>();
      // new bias = fc bias + gru bias
      out_bias_tensor->Resize(gru_bias_tenosr.dims());
      auto* data = out_bias_tensor->mutable_data<float>(platform::CPUPlace());
      for (int i = 0; i < out_bias_tensor->numel(); i++) {
        data[i] =
            fc_bias_tensor.data<float>()[i] + gru_bias_tenosr.data<float>()[i];
      }
    }
#undef GET_NODE

#define NEW_IMTERMEDIATE_OUT(key) \
  scope->Var(NEW_NAME(key))->GetMutable<framework::LoDTensor>()
    NEW_IMTERMEDIATE_OUT(ReorderedH0);
    NEW_IMTERMEDIATE_OUT(XX);
    NEW_IMTERMEDIATE_OUT(BatchedInput);
    NEW_IMTERMEDIATE_OUT(BatchedOut);
#undef NEW_NAME
#undef NEW_IMTERMEDIATE_OUT

    IR_NODE_LINK_TO(x_n, op);
    IR_NODE_LINK_TO(weight_x_n, op);
    IR_NODE_LINK_TO(weight_h_n, op);
    IR_NODE_LINK_TO(bias_n, op);  // actually should link to new bias if have
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
    GET_NODE(w);  // fc weight
    GET_NODE(mul);
    GET_NODE(fc_out);
    GET_NODE(Weight);
    GET_NODE(gru);
    GET_NODE(Bias);
    GET_NODE(Hidden);
    // nodes need be removed
    GET_NODE(BatchGate);
    GET_NODE(BatchResetHiddenPrev);
    GET_NODE(BatchHidden);

    if (with_fc_bias) {
      GET_NODE(mul_out);
      GET_NODE(fc_bias);
      GET_NODE(elementwise_add);
      gru_creater(gru, x, w, Weight, Bias, Hidden, fc_bias);
      // Remove unneeded nodes.
      std::unordered_set<const Node*> marked_nodes(
          {mul_n, gru_n, elementwise_add_n, fc_bias_n, fc_out_n, mul_out_n,
           BatchGate_n, BatchResetHiddenPrev_n, BatchHidden_n});
      GraphSafeRemoveNodes(graph, marked_nodes);
    } else {
      gru_creater(gru, x, w, Weight, Bias, Hidden, -1);
      // Remove unneeded nodes.
      std::unordered_set<const Node*> marked_nodes(
          {mul_n, gru_n, BatchGate_n, BatchResetHiddenPrev_n, BatchHidden_n});
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

REGISTER_PASS(mul_gru_fuse_pass, paddle::framework::ir::MulGRUFusePass);
REGISTER_PASS(fc_gru_fuse_pass, paddle::framework::ir::FCGRUFusePass);
