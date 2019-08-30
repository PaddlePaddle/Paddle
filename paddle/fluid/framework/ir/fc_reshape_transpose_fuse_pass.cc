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

#include "paddle/fluid/framework/ir/fc_reshape_transpose_fuse_pass.h"
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
                       Scope* scope) {
  GraphPatternDetector gpd;
  auto* pattern = gpd.mutable_pattern();

  // Create pattern.
  FCReshapeTransposePattern fc_rts_pattern(pattern, name_scope);

  PDNode* x =
      pattern->NewNode(patterns::UniqueKey("X"))->assert_var_not_persistable();

  fc_rts_pattern(x);
  // Create New OpDesc
  auto fuse_creater = [&](
      Node* x, Node* mul, Node* mul_out, Node* mul0_w, Node* mul1_w,
      Node* mul2_w, Node* eltadd, Node* eltadd_out, Node* eltadd0_b,
      Node* eltadd1_b, Node* eltadd2_b, Node* reshape2, Node* reshape2_out,
      Node* transpose2, Node* transpose2_out, Node* transpose2_1_out,
      Node* transpose2_2_out, Node* scale, Node* scale_out) {
    PADDLE_ENFORCE(graph->Has(kParamScopeAttr));

    auto* w0_var = scope->FindVar(mul0_w->Name());
    auto* w0_tensor = w0_var->GetMutable<framework::LoDTensor>();

    auto* w1_var = scope->FindVar(mul1_w->Name());
    auto* w1_tensor = w1_var->GetMutable<framework::LoDTensor>();

    auto* w2_var = scope->FindVar(mul2_w->Name());
    auto* w2_tensor = w2_var->GetMutable<framework::LoDTensor>();

    auto scale_attr = boost::get<float>(scale->Op()->GetAttr("scale"));
    auto scale_bias = boost::get<float>(scale->Op()->GetAttr("bias"));
    bool after_scale =
        boost::get<bool>(scale->Op()->GetAttr("bias_after_scale"));

    std::vector<int> w_init_dim =
        paddle::framework::vectorize2int(w0_tensor->dims());
    w_init_dim[1] *= 3;

    VarDesc fuse_column_w_desc(patterns::PDNodeName(name_scope, "Weight"));
    fuse_column_w_desc.SetPersistable(true);
    auto* fuse_column_w_node = graph->CreateVarNode(&fuse_column_w_desc);
    auto* fuse_column_w_tensor =
        scope->Var(fuse_column_w_node->Name())->GetMutable<LoDTensor>();
    fuse_column_w_tensor->Resize(framework::make_ddim({w_init_dim}));

    auto* data =
        fuse_column_w_tensor->mutable_data<float>(platform::CPUPlace());
    auto* w0_data = w0_tensor->mutable_data<float>(platform::CPUPlace());
    // scale for w0
    for (int i = 0; i < w0_tensor->numel(); ++i) {
      if (after_scale) {
        auto v = w0_data[i];
        w0_data[i] = v * scale_attr + scale_bias;
      } else {
        w0_data[i] = scale_attr * (w0_data[i] + scale_bias);
      }
    }
    auto* w1_data = w1_tensor->mutable_data<float>(platform::CPUPlace());
    auto* w2_data = w2_tensor->mutable_data<float>(platform::CPUPlace());

    int csize = w_init_dim[1] / 3;
    auto cb_size = csize * sizeof(float);

    for (int i = 0; i < w_init_dim[0]; ++i) {
      memcpy(data, w0_data, cb_size);
      data += csize;
      w0_data += csize;
      memcpy(data, w1_data, cb_size);
      data += csize;
      w1_data += csize;
      memcpy(data, w2_data, cb_size);
      data += csize;
      w2_data += csize;
    }

    auto* eltadd0_var = scope->FindVar(eltadd0_b->Name());
    auto* eltadd0_tensor = eltadd0_var->GetMutable<framework::LoDTensor>();
    auto* eltadd1_var = scope->FindVar(eltadd1_b->Name());
    auto* eltadd1_tensor = eltadd1_var->GetMutable<framework::LoDTensor>();
    auto* eltadd2_var = scope->FindVar(eltadd2_b->Name());
    auto* eltadd2_tensor = eltadd2_var->GetMutable<framework::LoDTensor>();

    std::vector<int> b_init_dim =
        paddle::framework::vectorize2int(eltadd0_tensor->dims());
    b_init_dim[0] *= 3;

    VarDesc fuse_column_bias_desc(patterns::PDNodeName(name_scope, "Bias"));
    fuse_column_bias_desc.SetPersistable(true);
    auto* fuse_column_bias_node = graph->CreateVarNode(&fuse_column_bias_desc);
    auto* fuse_column_bias_tensor =
        scope->Var(fuse_column_bias_node->Name())->GetMutable<LoDTensor>();
    fuse_column_bias_tensor->Resize(framework::make_ddim({b_init_dim}));

    auto* b_data =
        fuse_column_bias_tensor->mutable_data<float>(platform::CPUPlace());
    auto* b0_data = eltadd0_tensor->mutable_data<float>(platform::CPUPlace());
    // scale for bias
    for (int i = 0; i < eltadd0_tensor->numel(); ++i) {
      b0_data[i] = b0_data[i] * scale_attr;
    }
    auto* b1_data = eltadd1_tensor->mutable_data<float>(platform::CPUPlace());
    auto* b2_data = eltadd2_tensor->mutable_data<float>(platform::CPUPlace());

    auto bsize = b_init_dim[0] / 3;
    auto copy_size = bsize * sizeof(float);
    memcpy(b_data, b0_data, copy_size);
    memcpy(b_data + bsize, b1_data, copy_size);
    memcpy(b_data + 2 * bsize, b2_data, copy_size);

    mul->Op()->SetInput("X", {x->Name()});
    mul->Op()->SetInput("Y", {fuse_column_w_node->Name()});
    mul->Op()->SetOutput("Out", {mul_out->Name()});
    IR_NODE_LINK_TO(fuse_column_w_node, mul);

    eltadd->Op()->SetInput("X", {mul_out->Name()});
    eltadd->Op()->SetInput("Y", {fuse_column_bias_node->Name()});
    eltadd->Op()->SetOutput("Out", {eltadd_out->Name()});
    IR_NODE_LINK_TO(fuse_column_bias_node, eltadd);

    reshape2->Op()->SetInput("X", {eltadd_out->Name()});
    reshape2->Op()->SetOutput("Out", {reshape2_out->Name()});
    std::vector<int> shape =
        boost::get<std::vector<int>>(reshape2->Op()->GetAttr("shape"));
    shape[2] *= 3;
    reshape2->Op()->SetAttr("shape", shape);

    transpose2->Op()->SetInput("X", {reshape2_out->Name()});
    transpose2->Op()->SetOutput("Out", {transpose2_out->Name()});

    OpDesc split_op_desc;

    split_op_desc.SetType("split");
    split_op_desc.SetInput("X", {transpose2_out->Name()});
    split_op_desc.SetOutput("Out", {scale_out->Name(), transpose2_1_out->Name(),
                                    transpose2_2_out->Name()});
    split_op_desc.SetAttr("num", 3);
    split_op_desc.SetAttr("axis", 1);

    auto* split = graph->CreateOpNode(&split_op_desc);
    IR_NODE_LINK_TO(transpose2_out, split);
    IR_NODE_LINK_TO(split, scale_out);
    IR_NODE_LINK_TO(split, transpose2_1_out);
    IR_NODE_LINK_TO(split, transpose2_2_out);
  };

  int fusion_count{0};
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(dropout_out, dropout_out, fc_rts_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(mul0, mul0, fc_rts_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul0_out, mul0_out, fc_rts_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul0_w, mul0_w, fc_rts_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_0, reshape2_0, fc_rts_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_0_out, reshape2_0_out, fc_rts_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_0, transpose2_0, fc_rts_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_0_out, transpose2_0_out,
                              fc_rts_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale, scale, fc_rts_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale_out, scale_out, fc_rts_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(mul1, mul1, fc_rts_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul1_out, mul1_out, fc_rts_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul1_w, mul1_w, fc_rts_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_1, reshape2_1, fc_rts_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_1_out, reshape2_1_out, fc_rts_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_1, transpose2_1, fc_rts_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_1_out, transpose2_1_out,
                              fc_rts_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(mul2, mul2, fc_rts_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul2_out, mul2_out, fc_rts_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul2_w, mul2_w, fc_rts_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_2, reshape2_2, fc_rts_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_2_out, reshape2_2_out, fc_rts_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_2, transpose2_2, fc_rts_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_2_out, transpose2_2_out,
                              fc_rts_pattern);

    // nodes need be removed
    GET_IR_NODE_FROM_SUBGRAPH(eltadd0, eltadd0, fc_rts_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd0_b, eltadd0_b, fc_rts_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd0_out, eltadd0_out, fc_rts_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(eltadd1, eltadd1, fc_rts_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd1_b, eltadd1_b, fc_rts_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd1_out, eltadd1_out, fc_rts_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(eltadd2, eltadd2, fc_rts_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd2_b, eltadd2_b, fc_rts_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd2_out, eltadd2_out, fc_rts_pattern);

    fuse_creater(dropout_out, mul0, mul0_out, mul0_w, mul1_w, mul2_w, eltadd0,
                 eltadd0_out, eltadd0_b, eltadd1_b, eltadd2_b, reshape2_0,
                 reshape2_0_out, transpose2_0, transpose2_0_out,
                 transpose2_1_out, transpose2_2_out, scale, scale_out);

    std::unordered_set<const Node*> marked_nodes({
        mul1,       mul2,           mul0_w,         mul1_w,       mul2_w,
        mul1_out,   mul2_out,       eltadd1,        eltadd2,      eltadd0_b,
        eltadd1_b,  eltadd2_b,      eltadd1_out,    eltadd2_out,  reshape2_1,
        reshape2_2, reshape2_1_out, reshape2_2_out, transpose2_1, transpose2_2,
        scale,
    });
    GraphSafeRemoveNodes(graph, marked_nodes);
    // Remove unneeded nodes.
    ++fusion_count;
  };
  gpd(graph, handler);

  return fusion_count;
}

PDNode* FCReshapeTransposePattern::operator()(
    paddle::framework::ir::PDNode* x) {
  // Create shared nodes.
  auto* dropout = pattern->NewNode(dropout_repr());

  auto* dropout_out_var = pattern->NewNode(dropout_out_repr());
  dropout_out_var->assert_is_op_input("mul");

  // First path with scale
  auto* mul0 = pattern->NewNode(mul0_repr())->assert_is_op("mul");
  auto* mul0_w_var = pattern->NewNode(mul0_w_repr())
                         ->AsInput()
                         ->assert_is_op_input("mul", "Y");
  auto* mul0_out_var =
      pattern->NewNode(mul0_out_repr())->assert_is_op_output("mul");

  decltype(mul0) eltadd0;
  decltype(mul0) eltadd0_b_var;
  decltype(mul0) eltadd0_out_var;

  mul0_out_var->AsIntermediate()->assert_is_op_input("elementwise_add");

  eltadd0 = pattern->NewNode(eltadd0_repr())->assert_is_op("elementwise_add");
  eltadd0_b_var = pattern->NewNode(eltadd0_b_repr())
                      ->AsInput()
                      ->assert_is_op_input("elementwise_add", "Y");

  eltadd0_out_var = pattern->NewNode(eltadd0_out_repr())
                        ->assert_is_op_output("elementwise_add");
  eltadd0_out_var->AsIntermediate()->assert_is_op_input("reshape2");

  auto* reshape2_0 =
      pattern->NewNode(reshape2_0_repr())->assert_is_op("reshape2");

  auto* reshape2_0_out_var =
      pattern->NewNode(reshape2_0_out_repr())->assert_is_op_output("reshape2");
  reshape2_0_out_var->AsIntermediate()->assert_is_op_input("transpose2");

  auto* transpose2_0 =
      pattern->NewNode(transpose2_0_repr())->assert_is_op("transpose2");
  auto* transpose2_0_out_var = pattern->NewNode(transpose2_0_out_repr())
                                   ->assert_is_op_output("transpose2");
  transpose2_0_out_var->AsIntermediate()->assert_is_op_input("scale");

  auto* scale = pattern->NewNode(scale_repr())->assert_is_op("scale");
  auto* scale_out_var =
      pattern->NewNode(scale_out_repr())->assert_is_op_output("scale");

  // Second path to matmul
  auto* mul1 = pattern->NewNode(mul1_repr())->assert_is_op("mul");
  auto* mul1_w_var = pattern->NewNode(mul1_w_repr())
                         ->AsInput()
                         ->assert_is_op_input("mul", "Y");
  auto* mul1_out_var =
      pattern->NewNode(mul1_out_repr())->assert_is_op_output("mul");

  decltype(mul1) eltadd1;
  decltype(mul1) eltadd1_b_var;
  decltype(mul1) eltadd1_out_var;

  mul1_out_var->AsIntermediate()->assert_is_op_input("elementwise_add");
  eltadd1 = pattern->NewNode(eltadd1_repr())->assert_is_op("elementwise_add");
  eltadd1_b_var = pattern->NewNode(eltadd1_b_repr())
                      ->AsInput()
                      ->assert_is_op_input("elementwise_add", "Y");

  eltadd1_out_var = pattern->NewNode(eltadd1_out_repr())
                        ->assert_is_op_output("elementwise_add");
  eltadd1_out_var->AsIntermediate()->assert_is_op_input("reshape2");

  auto* reshape2_1 =
      pattern->NewNode(reshape2_1_repr())->assert_is_op("reshape2");

  auto* reshape2_1_out_var =
      pattern->NewNode(reshape2_1_out_repr())->assert_is_op_output("reshape2");
  reshape2_1_out_var->AsIntermediate()->assert_is_op_input("transpose2");

  auto* transpose2_1 =
      pattern->NewNode(transpose2_1_repr())->assert_is_op("transpose2");
  auto* transpose2_1_out_var = pattern->NewNode(transpose2_1_out_repr())
                                   ->assert_is_op_output("transpose2");
  transpose2_1_out_var->AsIntermediate()->assert_is_op_input("matmul");

  auto* matmul0 = pattern->NewNode(matmul0_repr())->assert_is_op("matmul");
  auto* matmul0_out_var =
      pattern->NewNode(matmul0_out_repr())->assert_is_op_output("matmul");

  // Third path to matmul
  auto* mul2 = pattern->NewNode(mul2_repr())->assert_is_op("mul");
  auto* mul2_w_var = pattern->NewNode(mul2_w_repr())
                         ->AsInput()
                         ->assert_is_op_input("mul", "Y");
  auto* mul2_out_var =
      pattern->NewNode(mul2_out_repr())->assert_is_op_output("mul");

  decltype(mul2) eltadd2;
  decltype(mul2) eltadd2_b_var;
  decltype(mul2) eltadd2_out_var;

  mul2_out_var->AsIntermediate()->assert_is_op_input("elementwise_add");
  eltadd2 = pattern->NewNode(eltadd2_repr())->assert_is_op("elementwise_add");
  eltadd2_b_var = pattern->NewNode(eltadd2_b_repr())
                      ->AsInput()
                      ->assert_is_op_input("elementwise_add", "Y");

  eltadd2_out_var = pattern->NewNode(eltadd2_out_repr())
                        ->assert_is_op_output("elementwise_add");
  eltadd2_out_var->AsIntermediate()->assert_is_op_input("reshape2");

  auto* reshape2_2 =
      pattern->NewNode(reshape2_2_repr())->assert_is_op("reshape2");

  auto* reshape2_2_out_var =
      pattern->NewNode(reshape2_2_out_repr())->assert_is_op_output("reshape2");
  reshape2_2_out_var->AsIntermediate()->assert_is_op_input("transpose2");

  auto* transpose2_2 =
      pattern->NewNode(transpose2_2_repr())->assert_is_op("transpose2");
  auto* transpose2_2_out_var = pattern->NewNode(transpose2_2_out_repr())
                                   ->assert_is_op_output("transpose2");
  transpose2_2_out_var->AsIntermediate()->assert_is_op_input("matmul");

  auto* matmul1 = pattern->NewNode(matmul1_repr())->assert_is_op("matmul");
  auto* matmul1_out_var =
      pattern->NewNode(matmul1_out_repr())->assert_is_op_output("matmul");

  // Link all nodes together
  dropout->LinksFrom({x}).LinksTo({dropout_out_var});
  mul0->LinksFrom({dropout_out_var, mul0_w_var}).LinksTo({mul0_out_var});

  eltadd0->LinksFrom({mul0_out_var, eltadd0_b_var}).LinksTo({eltadd0_out_var});

  reshape2_0->LinksFrom({eltadd0_out_var}).LinksTo({reshape2_0_out_var});
  transpose2_0->LinksFrom({reshape2_0_out_var}).LinksTo({transpose2_0_out_var});
  scale->LinksFrom({transpose2_0_out_var}).LinksTo({scale_out_var});

  mul1->LinksFrom({dropout_out_var, mul1_w_var}).LinksTo({mul1_out_var});

  eltadd1->LinksFrom({mul1_out_var, eltadd1_b_var}).LinksTo({eltadd1_out_var});

  reshape2_1->LinksFrom({eltadd1_out_var}).LinksTo({reshape2_1_out_var});
  transpose2_1->LinksFrom({reshape2_1_out_var}).LinksTo({transpose2_1_out_var});

  matmul0->LinksFrom({transpose2_1_out_var, scale_out_var})
      .LinksTo({matmul0_out_var});

  mul2->LinksFrom({dropout_out_var, mul2_w_var}).LinksTo({mul2_out_var});

  eltadd2->LinksFrom({mul2_out_var, eltadd2_b_var}).LinksTo({eltadd2_out_var});

  reshape2_2->LinksFrom({eltadd2_out_var}).LinksTo({reshape2_2_out_var});
  transpose2_2->LinksFrom({reshape2_2_out_var}).LinksTo({transpose2_2_out_var});

  matmul1->LinksFrom({transpose2_2_out_var}).LinksTo({matmul1_out_var});
  return transpose2_2_out_var;
}

}  // namespace patterns

void FCReshapeTransposeFusePass::ApplyImpl(Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);

  int fusion_count = patterns::BuildFusion(graph, name_scope_, param_scope());

  AddStatis(fusion_count);
  // return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fc_reshape_transpose_fuse_pass,
              paddle::framework::ir::FCReshapeTransposeFusePass);
