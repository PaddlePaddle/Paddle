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

#include "paddle/fluid/framework/ir/conv_bn_fuse_pass.h"
#include <functional>
#include <string>
#include <vector>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/operators/math/cpu_vec.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

LoDTensor tensor_apply(const LoDTensor& vec, float (*f)(float)) {
  LoDTensor vec_y;
  vec_y.Resize(vec.dims());
  const float* x = vec.data<float>();
  float* y = vec_y.mutable_data<float>(platform::CPUPlace());
  for (int i = 0; i < vec.numel(); i++) {
    y[i] = f(x[i]);
  }
  return vec_y;
}

void tensor_apply_inplace(LoDTensor& vec, float (*f)(float)) {
  float* data = vec.mutable_data<float>(platform::CPUPlace());
  for (int i = 0; i < vec.numel(); i++) {
    data[i] = f(data[i]);
  }
}

template <typename BinaryOperation>
LoDTensor tensor_apply_eltwise(const LoDTensor& vec_a, const LoDTensor& vec_b,
                               BinaryOperation f) {
  PADDLE_ENFORCE_EQ(vec_a.dims(), vec_b.dims());
  LoDTensor vec_y;
  vec_y.Resize(vec_a.dims());
  const float* a = vec_a.data<float>();
  const float* b = vec_b.data<float>();
  float* y = vec_y.mutable_data<float>(platform::CPUPlace());
  for (int i = 0; i < vec_a.numel(); i++) {
    y[i] = f(a[i], b[i]);
  }
  return vec_y;
}

template <typename BinaryOperation>
LoDTensor tensor_apply_eltwise_broadcast(const LoDTensor& vec_a,
                                         const LoDTensor& vec_b,
                                         BinaryOperation f) {
  PADDLE_ENFORCE_EQ(vec_a.dims().size(), 2);
  PADDLE_ENFORCE_EQ(vec_b.dims().size(), 2);
  PADDLE_ENFORCE_EQ(vec_a.dims()[0], vec_b.dims()[0]);
  PADDLE_ENFORCE_EQ(vec_b.dims()[1], 1);
  LoDTensor vec_y;
  vec_y.Resize(vec_a.dims());
  const float* a = vec_a.data<float>();
  const float* b = vec_b.data<float>();
  float* y = vec_y.mutable_data<float>(platform::CPUPlace());
  size_t a_height = vec_a.dims()[0];
  size_t a_width = vec_a.dims()[1];
  for (size_t h = 0; h < a_height; h++) {
    for (size_t w = 0; w < a_width; ++w) {
      *(y++) = f(*(a++), b[h]);
    }
  }
  return vec_y;
}

// reshape to two dimensions {A, B * C * ...}
void make_tensor_2d(LoDTensor& tensor_to_reshape) {
  auto dims_count = tensor_to_reshape.dims().size();
  PADDLE_ENFORCE_GT(dims_count, 0);

  int size2 = 1;
  for (int i = 1; i < dims_count; i++) {
    size2 *= tensor_to_reshape.dims()[i];
  }
  tensor_to_reshape.Resize(make_ddim({tensor_to_reshape.dims()[0], size2}));
}

std::unique_ptr<ir::Graph> ConvBNFusePass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  PADDLE_ENFORCE(graph.get());
  FusePassBase::Init(name_scope_, graph.get());

  auto* scope = param_scope();
  PADDLE_ENFORCE(scope);

  std::unordered_set<Node*> nodes2delete;

  GraphPatternDetector gpd;
  auto* conv_input =
      gpd.mutable_pattern()
          ->NewNode(patterns::PDNodeName(name_scope_, "conv_input"))
          ->AsInput()
          ->assert_is_op_input("conv2d", "Input");
  patterns::ConvBN conv_bn_pattern(gpd.mutable_pattern(), name_scope_);
  conv_bn_pattern(conv_input, false /*with_eltwise_add*/);

  int found_conv_bn_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "handle ConvBN fuse";
    // OPERATORS
    GET_IR_NODE_FROM_SUBGRAPH(conv, conv, conv_bn_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(batch_norm, batch_norm, conv_bn_pattern);
    // CONV inputs
    GET_IR_NODE_FROM_SUBGRAPH(conv_weight, conv_weight, conv_bn_pattern);
    // CONV outputs
    GET_IR_NODE_FROM_SUBGRAPH(conv_out, conv_out, conv_bn_pattern);
    // BN inputs
    GET_IR_NODE_FROM_SUBGRAPH(bn_scale, bn_scale, conv_bn_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(bn_bias, bn_bias, conv_bn_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(bn_mean, bn_mean, conv_bn_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(bn_variance, bn_variance, conv_bn_pattern);
    // BN outputs
    GET_IR_NODE_FROM_SUBGRAPH(bn_out, bn_out, conv_bn_pattern);  // Out
    GET_IR_NODE_FROM_SUBGRAPH(bn_mean_out, bn_mean_out, conv_bn_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(bn_variance_out, bn_variance_out,
                              conv_bn_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(bn_saved_mean, bn_saved_mean, conv_bn_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(bn_saved_variance, bn_saved_variance,
                              conv_bn_pattern);

    // Create y (bias) variable
    VarDesc y_desc(patterns::PDNodeName(name_scope_, "eltwise_y_in"));
    auto* y_var_node = g->CreateVarNode(&y_desc);
    auto* bias_y_tensor =
        scope->Var(y_var_node->Name())->GetMutable<LoDTensor>();
    std::fill_n(bias_y_tensor->mutable_data<float>(platform::CPUPlace()),
                bias_y_tensor->numel(), 0.0f);

    // Compute bias from BN
    auto* bn_bias_var = scope->FindVar(bn_bias->Name());
    auto* bn_bias_tensor = bn_bias_var->GetMutable<LoDTensor>();

    bias_y_tensor->Resize(bn_bias_tensor->dims());
    std::fill_n(bias_y_tensor->mutable_data<float>(platform::CPUPlace()),
                bn_bias_tensor->numel(), 0.0f);

    auto* scale_tensor =
        scope->FindVar(bn_scale->Name())->GetMutable<LoDTensor>();
    auto* variance_tensor =
        scope->FindVar(bn_variance->Name())->GetMutable<LoDTensor>();
    auto* mean_tensor =
        scope->FindVar(bn_mean->Name())->GetMutable<LoDTensor>();

    auto std_tensor = LoDTensor();
    std_tensor.Resize(bn_bias_tensor->dims());
    std_tensor =
        tensor_apply(*variance_tensor, [](float x) { return x + 1e-5f; });

    tensor_apply_inplace(std_tensor, std::sqrt);
    auto tmp_tensor =
        tensor_apply_eltwise(*scale_tensor, std_tensor, std::divides<float>());
    *bias_y_tensor = tensor_apply_eltwise(
        tensor_apply_eltwise(tensor_apply_eltwise(*bias_y_tensor, *mean_tensor,
                                                  std::minus<float>()),
                             tmp_tensor, std::multiplies<float>()),
        *bn_bias_tensor, std::plus<float>());

    // Re-compute weight of conv2d from BN
    auto* current_param =
        scope->FindVar(conv_weight->Name())->GetMutable<LoDTensor>();
    // remember the weights tensor shape {A, B, C, ...}
    auto current_param_shape = current_param->dims();
    // reduce the weights to 2d {A, B * C * ...}
    make_tensor_2d(*current_param);
    // make tmp tensor 2d by adding 1 as second dim {A, 1}
    make_tensor_2d(tmp_tensor);

    *current_param = tensor_apply_eltwise_broadcast(*current_param, tmp_tensor,
                                                    std::multiplies<float>());
    // reshape weights to the original dims {A, B, C, ...}
    current_param->Resize(current_param_shape);

    // Create an elementwise add node.
    OpDesc desc;
    desc.SetInput("X", std::vector<std::string>({conv_out->Name()}));
    desc.SetInput("Y", std::vector<std::string>({y_var_node->Name()}));
    desc.SetOutput("Out", std::vector<std::string>({bn_out->Name()}));
    desc.SetType("elementwise_add");
    desc.SetAttr("axis", 1);
    bool a = boost::get<bool>(conv->Op()->GetAttr("use_mkldnn"));
    desc.SetAttr("use_mkldnn", a);
    auto bias_op = g->CreateOpNode(&desc);  // OpDesc will be copied.

    GraphSafeRemoveNodes(graph.get(), {bn_scale, bn_bias, bn_mean, bn_variance,
                                       batch_norm, bn_mean_out, bn_variance_out,
                                       bn_saved_mean, bn_saved_variance});

    PADDLE_ENFORCE(subgraph.count(conv_input));
    IR_NODE_LINK_TO(conv_out, bias_op);
    IR_NODE_LINK_TO(y_var_node, bias_op);
    IR_NODE_LINK_TO(bias_op, bn_out);

    found_conv_bn_count++;
  };

  gpd(graph.get(), handler);

  AddStatis(found_conv_bn_count);
  return graph;
}

std::unique_ptr<ir::Graph> ConvEltwiseAddBNFusePass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  PADDLE_ENFORCE(graph.get());
  FusePassBase::Init(name_scope_, graph.get());

  auto* scope = param_scope();
  PADDLE_ENFORCE(scope);

  std::unordered_set<Node*> nodes2delete;

  GraphPatternDetector gpd;
  auto* conv_input =
      gpd.mutable_pattern()
          ->NewNode(patterns::PDNodeName(name_scope_, "conv_input"))
          ->AsInput()
          ->assert_is_op_input("conv2d", "Input");
  patterns::ConvBN conv_bn_pattern(gpd.mutable_pattern(), name_scope_);
  conv_bn_pattern(conv_input, true /*with_eltwise_add*/);

  int found_conv_bn_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "handle ConvBN fuse";
    // OPERATORS
    GET_IR_NODE_FROM_SUBGRAPH(conv, conv, conv_bn_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(bias, bias, conv_bn_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(batch_norm, batch_norm, conv_bn_pattern);
    // CONV inputs
    GET_IR_NODE_FROM_SUBGRAPH(conv_weight, conv_weight, conv_bn_pattern);
    // CONV outputs
    GET_IR_NODE_FROM_SUBGRAPH(conv_out, conv_out, conv_bn_pattern);
    // BIAS inputs
    GET_IR_NODE_FROM_SUBGRAPH(bias_y, bias_y, conv_bn_pattern);
    // BIAS outputs
    GET_IR_NODE_FROM_SUBGRAPH(bias_out, bias_out, conv_bn_pattern);
    // BN inputs
    GET_IR_NODE_FROM_SUBGRAPH(bn_scale, bn_scale, conv_bn_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(bn_bias, bn_bias, conv_bn_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(bn_mean, bn_mean, conv_bn_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(bn_variance, bn_variance, conv_bn_pattern);
    // BN outputs
    GET_IR_NODE_FROM_SUBGRAPH(bn_out, bn_out, conv_bn_pattern);  // Out
    GET_IR_NODE_FROM_SUBGRAPH(bn_mean_out, bn_mean_out, conv_bn_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(bn_variance_out, bn_variance_out,
                              conv_bn_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(bn_saved_mean, bn_saved_mean, conv_bn_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(bn_saved_variance, bn_saved_variance,
                              conv_bn_pattern);

    // Find bias from BN
    auto* bn_bias_var = scope->FindVar(bn_bias->Name());
    auto* bn_bias_tensor = bn_bias_var->GetMutable<LoDTensor>();

    // Get y (bias) variable
    auto* bias_y_tensor =
        scope->FindVar(bias_y->Name())->GetMutable<LoDTensor>();

    PADDLE_ENFORCE_EQ(bias_y_tensor->dims(), bn_bias_tensor->dims());

    // Compute new weights and bias
    auto* scale_tensor =
        scope->FindVar(bn_scale->Name())->GetMutable<LoDTensor>();
    auto* variance_tensor =
        scope->FindVar(bn_variance->Name())->GetMutable<LoDTensor>();
    auto* mean_tensor =
        scope->FindVar(bn_mean->Name())->GetMutable<LoDTensor>();

    auto std_tensor = LoDTensor();
    std_tensor.Resize(bn_bias_tensor->dims());
    std_tensor =
        tensor_apply(*variance_tensor, [](float x) { return x + 1e-5f; });

    tensor_apply_inplace(std_tensor, std::sqrt);
    auto tmp_tensor =
        tensor_apply_eltwise(*scale_tensor, std_tensor, std::divides<float>());
    *bias_y_tensor = tensor_apply_eltwise(
        tensor_apply_eltwise(tensor_apply_eltwise(*bias_y_tensor, *mean_tensor,
                                                  std::minus<float>()),
                             tmp_tensor, std::multiplies<float>()),
        *bn_bias_tensor, std::plus<float>());

    // Re-compute weight of conv2d from BN
    auto* current_param =
        scope->FindVar(conv_weight->Name())->GetMutable<LoDTensor>();
    // remember the weights tensor shape {A, B, C, ...}
    auto current_param_shape = current_param->dims();
    // flatten the weights tensor shape {A, B * C * ...}
    if (current_param->dims().size() > 1) {
      int size2 = current_param->dims()[1];
      for (int i = 2; i < current_param->dims().size(); i++) {
        size2 *= current_param->dims()[i];
      }
      current_param->Resize(make_ddim({current_param->dims()[0], size2}));
    }
    *current_param = tensor_apply_eltwise_broadcast(*current_param, tmp_tensor,
                                                    std::multiplies<float>());
    // reshape weights to the original dims {A, B, C, ...}
    current_param->Resize(current_param_shape);

    // Modify bias of elementwise add node.
    bias->Op()->SetAttr("axis", 1);
    bias->Op()->SetOutput("Out", std::vector<std::string>({bn_out->Name()}));

    GraphSafeRemoveNodes(
        graph.get(),
        {bn_scale, bn_bias, bn_mean, bn_variance, batch_norm, bn_mean_out,
         bn_variance_out, bn_saved_mean, bn_saved_variance, bias_out});

    PADDLE_ENFORCE(subgraph.count(conv_input));
    IR_NODE_LINK_TO(bias, bn_out);

    found_conv_bn_count++;
  };

  gpd(graph.get(), handler);

  AddStatis(found_conv_bn_count);
  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(conv_bn_fuse_pass, paddle::framework::ir::ConvBNFusePass);
REGISTER_PASS(conv_eltwiseadd_bn_fuse_pass,
              paddle::framework::ir::ConvEltwiseAddBNFusePass);
