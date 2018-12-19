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

#include "paddle/fluid/framework/ir/fc_cudnn_gru_fuse_pass.h"
#include <string>
#include <vector>
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/lod_tensor.h"

namespace paddle {
namespace framework {
namespace ir {

static void CopyWeight(Tensor* w_tensor, Tensor* weight_x_tensor,
                       Tensor* weight_h_tensor, Tensor* gru_bias_tensor,
                       Scope* scope, Node* fc_bias, bool with_fc_bias) {
  float* w_data = w_tensor->data<float>();
  // copy wx data
  int offset = weight_x_tensor->dims()[0] * weight_h_tensor->dims()[0];

  memcpy(w_data, weight_x_tensor->data<float>() + offset,
         offset * sizeof(float));
  w_data += offset;

  // copy update gate data
  memcpy(w_data, weight_x_tensor->data<float>(), offset * sizeof(float));
  w_data += offset;

  memcpy(w_data, weight_x_tensor->data<float>() + 2 * offset,
         offset * sizeof(float));
  w_data += offset;

  // copy wh data
  offset = weight_h_tensor->dims()[0] * weight_h_tensor->dims()[0];

  memcpy(w_data, weight_h_tensor->data<float>() + offset,
         offset * sizeof(float));
  w_data += offset;

  memcpy(w_data, weight_h_tensor->data<float>(), offset * sizeof(float));
  w_data += offset;

  memcpy(w_data, weight_h_tensor->data<float>() + offset * 2,
         offset * sizeof(float));
  w_data += offset;

  // copy bias data
  offset = gru_bias_tensor->dims()[1] / 3;
  if (with_fc_bias) {
    auto* fc_bias_var = scope->FindVar(fc_bias->Name());
    PADDLE_ENFORCE(fc_bias_var);
    auto* fc_bias_tensor = fc_bias_var->GetMutable<framework::LoDTensor>();

    memcpy(w_data, fc_bias_tensor->data<float>() + offset,
           offset * sizeof(float));
    w_data += offset;

    memcpy(w_data, fc_bias_tensor->data<float>(), offset * sizeof(float));
    w_data += offset;

    memcpy(w_data, fc_bias_tensor->data<float>() + offset * 2,
           offset * sizeof(float));
    w_data += offset;
  } else {
    // bx all 0
    memset(w_data, 0, offset * 3 * sizeof(float));
    w_data += offset * 3;
  }

  // copy bh
  memcpy(w_data, gru_bias_tensor->data<float>() + offset,
         offset * sizeof(float));
  w_data += offset;

  memcpy(w_data, gru_bias_tensor->data<float>(), offset * sizeof(float));
  w_data += offset;

  memcpy(w_data, gru_bias_tensor->data<float>() + offset * 2,
         offset * sizeof(float));
  w_data += offset;
}
static int BuildFusion(Graph* graph, const std::string& name_scope,
                       Scope* scope, bool with_fc_bias) {
  GraphPatternDetector gpd;
  auto* pattern = gpd.mutable_pattern();

  // Create pattern.
  patterns::FC fc_pattern(pattern, name_scope);
  patterns::GRU gru_pattern(pattern, name_scope);

  PDNode* x =
      pattern->NewNode(patterns::UniqueKey("x"))->assert_var_not_persistable();

  auto* fc_out = fc_pattern(x, with_fc_bias);
  // fc_out is a tmp var, will be removed after fuse.
  fc_out->AsIntermediate();
  gru_pattern(fc_out);
  // Create New OpDesc
  auto gru_creater = [&](Node* gru, Node* x, Node* weight_x, Node* weight_h,
                         Node* bias, Node* hidden, Node* fc_bias) {
    OpDesc op_desc;
    op_desc.SetType("cudnn_gru");
#define NEW_NAME(x) name_scope + "/at." #x ".new"
#define SET_IN(Key, node__) op_desc.SetInput(#Key, {node__->Name()});
    SET_IN(Input, x);
#undef SET_IN
    op_desc.SetOutput("Out", {hidden->Name()});

    PADDLE_ENFORCE(graph->Has(kParamScopeAttr));
    auto* scope = graph->Get<Scope*>(kParamScopeAttr);

    const std::string CacheNode = patterns::UniqueKey("Cache");

    VarDesc cudnn_gru_w_desc(patterns::PDNodeName(name_scope, "W"));
    cudnn_gru_w_desc.SetPersistable(true);
    auto* cudnn_gru_w_node = graph->CreateVarNode(&cudnn_gru_w_desc);
    auto* w_tensor =
        scope->Var(cudnn_gru_w_node->Name())->GetMutable<LoDTensor>();

    // create a inith node wizh size=0
    VarDesc cudnn_gru_inith_desc(patterns::PDNodeName(name_scope, "InitH"));
    cudnn_gru_inith_desc.SetPersistable(true);
    auto* cudnn_gru_inith_node = graph->CreateVarNode(&cudnn_gru_inith_desc);
    auto* h_tensor =
        scope->Var(cudnn_gru_inith_node->Name())->GetMutable<LoDTensor>();
    std::vector<int64_t> ha = {0, 0};
    h_tensor->Resize(framework::make_ddim(ha));
    std::fill_n(h_tensor->mutable_data<float>(platform::CPUPlace()),
                h_tensor->numel(), 0.0f);

    scope->Var(CacheNode);
    op_desc.SetInput("Cache", {CacheNode});

    op_desc.SetInput("W", {cudnn_gru_w_node->Name()});

    op_desc.SetInput("InitH", {cudnn_gru_inith_node->Name()});

    auto* weight_x_var = scope->FindVar(weight_x->Name());
    auto* weight_h_var = scope->FindVar(weight_h->Name());

    auto* weight_x_tensor = weight_x_var->GetMutable<framework::LoDTensor>();
    auto* weight_h_tensor = weight_h_var->GetMutable<framework::LoDTensor>();
    auto* gru_bias_var = scope->FindVar(bias->Name());
    auto* gru_bias_tensor = gru_bias_var->GetMutable<framework::LoDTensor>();

    int input_size = weight_x_tensor->dims()[0];
    int hidden_size = weight_h_tensor->dims()[0];

    weight_x_tensor->mutable_data<float>(platform::CPUPlace());
    weight_h_tensor->mutable_data<float>(platform::CPUPlace());
    gru_bias_tensor->mutable_data<float>(platform::CPUPlace());

    op_desc.SetAttr("input_size", input_size);
    op_desc.SetAttr("hidden_size", hidden_size);
    op_desc.SetAttr("max_len", 64);

    std::vector<int64_t> wdim = {weight_x_tensor->dims()[0] * 3 +
                                     weight_h_tensor->dims()[1] +
                                     gru_bias_tensor->dims()[0] * 6,
                                 weight_h_tensor->dims()[0]};

    framework::DDim dim = framework::make_ddim(wdim);
    w_tensor->Resize(dim);
    w_tensor->mutable_data<float>(platform::CPUPlace());
    // copy reset gate furst
    CopyWeight(w_tensor, weight_x_tensor, weight_h_tensor, gru_bias_tensor,
               scope, fc_bias, with_fc_bias);

    auto* op = graph->CreateOpNode(&op_desc);

    IR_NODE_LINK_TO(x, op);
    IR_NODE_LINK_TO(cudnn_gru_w_node, op);
    IR_NODE_LINK_TO(cudnn_gru_inith_node, op);
    IR_NODE_LINK_TO(op, hidden);
    return op;
  };

  int fusion_count{0};
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    auto* x_n = subgraph.at(x);
    GET_IR_NODE_FROM_SUBGRAPH(w, w, fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul, mul, fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_out, mul_out, fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(Weight, Weight, gru_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(gru, gru, gru_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(Bias, Bias, gru_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(Hidden, Hidden, gru_pattern);
    // nodes need be removed
    GET_IR_NODE_FROM_SUBGRAPH(BatchGate, BatchGate, gru_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(BatchResetHiddenPrev, BatchGate, gru_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(BatchHidden, BatchGate, gru_pattern);

    if (with_fc_bias) {
      GET_IR_NODE_FROM_SUBGRAPH(mul_out, mul_out, fc_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(fc_bias, bias, fc_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(elementwise_add, elementwise_add, fc_pattern);

      gru_creater(gru, x_n, w, Weight, Bias, Hidden, fc_bias);
      // Remove unneeded nodes.
      std::unordered_set<const Node*> marked_nodes(
          {mul, gru, elementwise_add, fc_bias, BatchGate, BatchResetHiddenPrev,
           BatchHidden});
      GraphSafeRemoveNodes(graph, marked_nodes);
    } else {
      gru_creater(gru, x_n, w, Weight, Bias, Hidden, nullptr);
      // Remove unneeded nodes.
      std::unordered_set<const Node*> marked_nodes(
          {mul, gru, Bias, Weight, w, BatchGate, BatchResetHiddenPrev,
           BatchHidden});
      GraphSafeRemoveNodes(graph, marked_nodes);
    }
    ++fusion_count;
  };
  gpd(graph, handler);

  return fusion_count;
}

std::unique_ptr<ir::Graph> MulCudnnGRUFusePass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  FusePassBase::Init(name_scope_, graph.get());

  int fusion_count = BuildFusion(graph.get(), name_scope_, param_scope(),
                                 false /*with_fc_bias*/);

  AddStatis(fusion_count);
  return graph;
}

std::unique_ptr<ir::Graph> FCCudnnGRUFusePass::ApplyImpl(
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

REGISTER_PASS(mul_cudnn_gru_fuse_pass,
              paddle::framework::ir::MulCudnnGRUFusePass);
REGISTER_PASS(fc_cudnn_gru_fuse_pass,
              paddle::framework::ir::FCCudnnGRUFusePass);
