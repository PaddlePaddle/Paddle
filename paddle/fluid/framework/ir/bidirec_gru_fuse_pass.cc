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

#include "paddle/fluid/framework/ir/bidirec_gru_fuse_pass.h"
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/lod_tensor.h"

namespace paddle {
namespace framework {
namespace ir {

static int BuildFusion(Graph* graph, const std::string& name_scope,
                       Scope* scope, bool with_fc_bias) {
  GraphPatternDetector gpd;
  auto* pattern = gpd.mutable_pattern();

  // Create pattern.
  patterns::BidirecGRU gru_pattern(pattern, name_scope);

  PDNode* x =
      pattern->NewNode(patterns::UniqueKey("X"))->assert_var_not_persistable();

  gru_pattern(x, with_fc_bias);
  // Create New OpDesc
  auto gru_creater = [&](Node* x, Node* mul0_w, Node* mul1_w, Node* mul2_w,
                         Node* mul3_w, Node* eltadd0_b, Node* eltadd1_b,
                         Node* gru0, Node* gru1, Node* weight0, Node* weight1,
                         Node* bias0, Node* bias1, Node* sum_out) {
    OpDesc op_desc;
    op_desc.SetType("fusion_bidirectional_gru");
#define NEW_NAME(x) name_scope + "/at." #x ".new"
#define SET_IN(Key, node__) op_desc.SetInput(#Key, {node__->Name()});
    SET_IN(X, x);
    SET_IN(Weight0X, mul0_w);
    SET_IN(Weight1X, mul1_w);
    SET_IN(Weight2X, mul2_w);
    SET_IN(Weight3X, mul3_w);
    SET_IN(Weight0H, weight0);
    SET_IN(Weight1H, weight1);
    if (with_fc_bias) {
      SET_IN(Bias0X, eltadd0_b);
      SET_IN(Bias1X, eltadd1_b);
    }
    SET_IN(Bias0H, bias0);
    SET_IN(Bias1H, bias1);
#undef SET_IN

#define SET_IMTERMEDIATE_OUT(key) op_desc.SetOutput(#key, {NEW_NAME(key)})
    SET_IMTERMEDIATE_OUT(mul_out0);
    SET_IMTERMEDIATE_OUT(mul_out1);
    SET_IMTERMEDIATE_OUT(gru_out0);
    SET_IMTERMEDIATE_OUT(gru_out1);
    SET_IMTERMEDIATE_OUT(gate0);
    SET_IMTERMEDIATE_OUT(gate1);
#undef SET_IMTERMEDIATE_OUT

    // Get gru bias dim as init dims
    auto* bias_var = scope->FindVar(bias0->Name());
    auto* bias_tensor = bias_var->GetMutable<framework::LoDTensor>();
    std::vector<int64_t> init_dim = {bias_tensor->dims()[0],
                                     bias_tensor->dims()[1] / 3};

    // Create default inith for gru op.
    VarDesc gru_inith0_desc(patterns::PDNodeName(name_scope, "InitH0"));
    gru_inith0_desc.SetPersistable(true);
    auto* gru_inith0_node = graph->CreateVarNode(&gru_inith0_desc);
    auto* h0_tensor =
        scope->Var(gru_inith0_node->Name())->GetMutable<LoDTensor>();
    h0_tensor->Resize(framework::make_ddim(init_dim));
    std::fill_n(h0_tensor->mutable_data<float>(platform::CPUPlace()),
                h0_tensor->numel(), 0.0f);

    VarDesc gru_inith1_desc(patterns::PDNodeName(name_scope, "InitH1"));
    gru_inith1_desc.SetPersistable(true);
    auto* gru_inith1_node = graph->CreateVarNode(&gru_inith1_desc);
    auto* h1_tensor =
        scope->Var(gru_inith1_node->Name())->GetMutable<LoDTensor>();
    h1_tensor->Resize(framework::make_ddim(init_dim));
    std::fill_n(h1_tensor->mutable_data<float>(platform::CPUPlace()),
                h1_tensor->numel(), 0.0f);

    op_desc.SetInput("InitH0", {gru_inith0_node->Name()});
    op_desc.SetInput("InitH1", {gru_inith1_node->Name()});

    op_desc.SetOutput("Out", {sum_out->Name()});

    PADDLE_ENFORCE(graph->Has(kParamScopeAttr));
    op_desc.SetAttr("gate_activation", gru0->Op()->GetAttr("gate_activation"));
    op_desc.SetAttr("activation", gru0->Op()->GetAttr("activation"));

    int reverse = 0;

    if (boost::get<bool>(gru0->Op()->GetAttr("is_reverse")))
      reverse = 0;
    else if (boost::get<bool>(gru1->Op()->GetAttr("is_reverse")))
      reverse = 1;

    op_desc.SetAttr("reverse", reverse);

    auto* op = graph->CreateOpNode(&op_desc);

#define NEW_IMTERMEDIATE_OUT(key)          \
  scope->Var(NEW_NAME(key))                \
      ->GetMutable<framework::LoDTensor>() \
      ->mutable_data<float>(platform::CPUPlace())
    NEW_IMTERMEDIATE_OUT(mul_out0);
    NEW_IMTERMEDIATE_OUT(mul_out1);
    NEW_IMTERMEDIATE_OUT(gru_out0);
    NEW_IMTERMEDIATE_OUT(gru_out1);
    NEW_IMTERMEDIATE_OUT(gate0);
    NEW_IMTERMEDIATE_OUT(gate1);
#undef NEW_NAME
#undef NEW_IMTERMEDIATE_OUT

    IR_NODE_LINK_TO(x, op);
    IR_NODE_LINK_TO(mul0_w, op);
    IR_NODE_LINK_TO(mul1_w, op);
    IR_NODE_LINK_TO(mul2_w, op);
    IR_NODE_LINK_TO(mul3_w, op);
    IR_NODE_LINK_TO(weight0, op);
    IR_NODE_LINK_TO(weight1, op);
    if (with_fc_bias) {
      IR_NODE_LINK_TO(eltadd0_b, op);
      IR_NODE_LINK_TO(eltadd1_b, op);
    }
    IR_NODE_LINK_TO(gru_inith0_node, op);
    IR_NODE_LINK_TO(gru_inith1_node, op);
    IR_NODE_LINK_TO(bias0, op);
    IR_NODE_LINK_TO(bias1, op);
    IR_NODE_LINK_TO(op, sum_out);
    return op;
  };

  int fusion_count{0};
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(xnode_out, xnode_out, gru_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(gru0, gru0, gru_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(Weight0, gru0_w, gru_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(Bias0, gru0_b, gru_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(hidden0, hidden0, gru_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(BatchGate0, BatchGate0, gru_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(BatchResetHiddenPrev0, BatchResetHiddenPrev0,
                              gru_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(BatchHidden0, BatchHidden0, gru_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(gru1, gru1, gru_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(Weight1, gru1_w, gru_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(Bias1, gru1_b, gru_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(hidden1, hidden1, gru_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(BatchGate1, BatchGate1, gru_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(BatchResetHiddenPrev1, BatchResetHiddenPrev1,
                              gru_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(BatchHidden1, BatchHidden1, gru_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(mul0, mul0, gru_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul0_out, mul0_out, gru_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul0_w, mul0_w, gru_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(mul1, mul1, gru_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul1_out, mul1_out, gru_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul1_w, mul1_w, gru_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(mul2, mul2, gru_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul2_out, mul2_out, gru_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul2_w, mul2_w, gru_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(mul3, mul3, gru_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul3_out, mul3_out, gru_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul3_w, mul3_w, gru_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(sum, sum, gru_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(sum_out, sum_out, gru_pattern);
    // nodes need be removed
    if (with_fc_bias) {
      GET_IR_NODE_FROM_SUBGRAPH(eltadd0, eltadd0, gru_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(eltadd0_b, eltadd0_b, gru_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(eltadd0_out, eltadd0_out, gru_pattern);

      GET_IR_NODE_FROM_SUBGRAPH(eltadd1, eltadd1, gru_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(eltadd1_b, eltadd1_b, gru_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(eltadd1_out, eltadd1_out, gru_pattern);
      gru_creater(xnode_out, mul0_w, mul1_w, mul2_w, mul3_w, eltadd0_b,
                  eltadd1_b, gru0, gru1, Weight0, Weight1, Bias0, Bias1,
                  sum_out);
      std::unordered_set<const Node*> marked_nodes({mul0,
                                                    mul1,
                                                    mul2,
                                                    mul3,
                                                    mul0_out,
                                                    mul1_out,
                                                    mul2_out,
                                                    mul3_out,
                                                    gru0,
                                                    gru1,
                                                    eltadd0,
                                                    eltadd1,
                                                    eltadd0_out,
                                                    eltadd1_out,
                                                    hidden0,
                                                    hidden1,
                                                    BatchGate0,
                                                    BatchGate1,
                                                    BatchResetHiddenPrev0,
                                                    BatchResetHiddenPrev1,
                                                    BatchHidden0,
                                                    BatchHidden1,
                                                    sum});
      GraphSafeRemoveNodes(graph, marked_nodes);
    } else {
      gru_creater(xnode_out, mul0_w, mul1_w, mul2_w, mul3_w, nullptr, nullptr,
                  gru0, gru1, Weight0, Weight1, Bias0, Bias1, sum_out);
      std::unordered_set<const Node*> marked_nodes(
          {mul0, mul1, mul2, mul3, mul0_out, mul1_out, mul2_out, mul3_out, gru0,
           gru1, hidden0, hidden1, BatchGate0, BatchGate1,
           BatchResetHiddenPrev0, BatchResetHiddenPrev1, BatchHidden0,
           BatchHidden1, sum});
      GraphSafeRemoveNodes(graph, marked_nodes);
    }
    // Remove unneeded nodes.
    ++fusion_count;
  };
  gpd(graph, handler);

  return fusion_count;
}

std::unique_ptr<ir::Graph> BidirecMulGRUFusePass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  FusePassBase::Init(name_scope_, graph.get());

  int fusion_count =
      BuildFusion(graph.get(), name_scope_, param_scope(), false);

  AddStatis(fusion_count);
  return graph;
}

std::unique_ptr<ir::Graph> BidirecFCGRUFusePass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  FusePassBase::Init(name_scope_, graph.get());
  int fusion_count = BuildFusion(graph.get(), name_scope_, param_scope(), true);

  AddStatis(fusion_count);
  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(bidirec_mul_gru_fuse_pass,
              paddle::framework::ir::BidirecMulGRUFusePass);
REGISTER_PASS(bidirec_gru_fuse_pass,
              paddle::framework::ir::BidirecFCGRUFusePass);
