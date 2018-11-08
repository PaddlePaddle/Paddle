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

#define GET_CONV_BN_NODES(pattern_name)                                      \
  /* OPERATORS */                                                            \
  GET_IR_NODE_FROM_SUBGRAPH(conv, conv, pattern_name);                       \
  GET_IR_NODE_FROM_SUBGRAPH(batch_norm, batch_norm, pattern_name);           \
  /* CONV inputs */                                                          \
  GET_IR_NODE_FROM_SUBGRAPH(conv_weight, conv_weight, pattern_name);         \
  /* CONV outputs */                                                         \
  GET_IR_NODE_FROM_SUBGRAPH(conv_out, conv_out, pattern_name);               \
  /* BN inputs */                                                            \
  GET_IR_NODE_FROM_SUBGRAPH(bn_scale, bn_scale, pattern_name);               \
  GET_IR_NODE_FROM_SUBGRAPH(bn_bias, bn_bias, pattern_name);                 \
  GET_IR_NODE_FROM_SUBGRAPH(bn_mean, bn_mean, pattern_name);                 \
  GET_IR_NODE_FROM_SUBGRAPH(bn_variance, bn_variance, pattern_name);         \
  /* BN outputs */                                                           \
  GET_IR_NODE_FROM_SUBGRAPH(bn_out, bn_out, pattern_name); /* Out */         \
  GET_IR_NODE_FROM_SUBGRAPH(bn_mean_out, bn_mean_out, pattern_name);         \
  GET_IR_NODE_FROM_SUBGRAPH(bn_variance_out, bn_variance_out, pattern_name); \
  GET_IR_NODE_FROM_SUBGRAPH(bn_saved_mean, bn_saved_mean, pattern_name);     \
  GET_IR_NODE_FROM_SUBGRAPH(bn_saved_variance, bn_saved_variance, pattern_name)

void recompute_bias_and_weights(const Scope* scope,
                                ir::Node* conv_weight,            //
                                const ir::Node& bn_scale,         //
                                const LoDTensor& bn_bias_tensor,  //
                                const ir::Node& bn_mean,          //
                                const ir::Node& bn_variance,      //
                                LoDTensor* eltwise_y_in_tensor,   //
                                float epsilon) {
  using EigenVectorArrayMap =
      Eigen::Map<Eigen::Array<float, Eigen::Dynamic, 1>>;
  using ConstEigenVectorArrayMap =
      Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, 1>>;
  using EigenMatrixArrayMap = Eigen::Map<
      Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

  // Re-compute bias of conv2d from BN
  PADDLE_ENFORCE_EQ(eltwise_y_in_tensor->dims(), bn_bias_tensor.dims());

  auto* scale_tensor = scope->FindVar(bn_scale.Name())->GetMutable<LoDTensor>();
  auto* variance_tensor =
      scope->FindVar(bn_variance.Name())->GetMutable<LoDTensor>();
  auto* mean_tensor = scope->FindVar(bn_mean.Name())->GetMutable<LoDTensor>();

  ConstEigenVectorArrayMap scale_array(scale_tensor->data<float>(),
                                       scale_tensor->numel(), 1);
  EigenVectorArrayMap variance_array(
      variance_tensor->mutable_data<float>(platform::CPUPlace()),
      variance_tensor->numel(), 1);
  ConstEigenVectorArrayMap mean_array(mean_tensor->data<float>(),
                                      mean_tensor->numel(), 1);
  ConstEigenVectorArrayMap bn_bias_array(bn_bias_tensor.data<float>(),
                                         bn_bias_tensor.numel(), 1);

  // variance will not be used anymore, so make it std_array and then tmp_array
  variance_array += epsilon;
  variance_array = variance_array.sqrt();
  variance_array = scale_array / variance_array;

  EigenVectorArrayMap eltwise_y_in_array(
      eltwise_y_in_tensor->mutable_data<float>(platform::CPUPlace()),
      eltwise_y_in_tensor->numel(), 1);

  eltwise_y_in_array =
      ((eltwise_y_in_array - mean_array) * variance_array) + bn_bias_array;

  // Re-compute weight of conv2d from BN
  auto* weights = scope->FindVar(conv_weight->Name())->GetMutable<LoDTensor>();
  auto weights_shape = weights->dims();
  auto weights_shape_2d = flatten_to_2d(weights_shape, 1);

  EigenMatrixArrayMap weights_array_2d(
      weights->mutable_data<float>(platform::CPUPlace()), weights_shape_2d[0],
      weights_shape_2d[1]);

  weights_array_2d.colwise() *= variance_array;
}

std::unique_ptr<ir::Graph> ConvBNFusePass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  PADDLE_ENFORCE(graph.get());
  FusePassBase::Init(name_scope_, graph.get());

  auto* scope = param_scope();
  PADDLE_ENFORCE(scope);

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
    VLOG(40) << "handle ConvBN fuse";

    // conv, batch_norm,
    // conv_weight, conv_out,
    // bn_scale, bn_bias, bn_mean, bn_variance,
    // bn_out, bn_mean_out, bn_variance_out, bn_saved_mean, bn_saved_variance
    GET_CONV_BN_NODES(conv_bn_pattern);

    // Create eltwise_y (conv bias) variable
    VarDesc eltwise_y_in_desc(
        patterns::PDNodeName(name_scope_, "eltwise_y_in"));
    auto* eltwise_y_in_node = g->CreateVarNode(&eltwise_y_in_desc);
    auto* eltwise_y_in_tensor =
        scope->Var(eltwise_y_in_node->Name())->GetMutable<LoDTensor>();

    // Get batch norm bias
    auto* bn_bias_tensor =
        scope->FindVar(bn_bias->Name())->GetMutable<LoDTensor>();

    // Initialize eltwise_y
    eltwise_y_in_tensor->Resize(bn_bias_tensor->dims());
    std::fill_n(eltwise_y_in_tensor->mutable_data<float>(platform::CPUPlace()),
                eltwise_y_in_tensor->numel(), 0.0f);

    // update weights and biases
    float epsilon = boost::get<float>(batch_norm->Op()->GetAttr("epsilon"));
    recompute_bias_and_weights(scope, conv_weight, *bn_scale, *bn_bias_tensor,
                               *bn_mean, *bn_variance, eltwise_y_in_tensor,
                               epsilon);

    // Create an elementwise add node
    OpDesc desc;
    desc.SetInput("X", std::vector<std::string>({conv_out->Name()}));
    desc.SetInput("Y", std::vector<std::string>({eltwise_y_in_node->Name()}));
    desc.SetOutput("Out", std::vector<std::string>({bn_out->Name()}));
    desc.SetType("elementwise_add");
    desc.SetAttr("axis", 1);
    bool a = boost::get<bool>(conv->Op()->GetAttr("use_mkldnn"));
    desc.SetAttr("use_mkldnn", a);
    auto eltwise_op = g->CreateOpNode(&desc);  // OpDesc will be copied.

    GraphSafeRemoveNodes(graph.get(), {bn_scale, bn_bias, bn_mean, bn_variance,
                                       batch_norm, bn_mean_out, bn_variance_out,
                                       bn_saved_mean, bn_saved_variance});

    PADDLE_ENFORCE(subgraph.count(conv_input));
    IR_NODE_LINK_TO(conv_out, eltwise_op);
    IR_NODE_LINK_TO(eltwise_y_in_node, eltwise_op);
    IR_NODE_LINK_TO(eltwise_op, bn_out);

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
    VLOG(40) << "handle ConvBN fuse";

    // conv, batch_norm,
    // conv_weight, conv_out,
    // bn_scale, bn_bias, bn_mean, bn_variance,
    // bn_out, bn_mean_out, bn_variance_out, bn_saved_mean,bn_saved_variance
    GET_CONV_BN_NODES(conv_bn_pattern);
    // OPERATORS
    GET_IR_NODE_FROM_SUBGRAPH(eltwise, eltwise, conv_bn_pattern);
    // BIAS inputs
    GET_IR_NODE_FROM_SUBGRAPH(eltwise_y_in, eltwise_y_in, conv_bn_pattern);
    // BIAS outputs
    GET_IR_NODE_FROM_SUBGRAPH(eltwise_out, eltwise_out, conv_bn_pattern);

    // Get eltwise_y (conv bias) variable
    auto* eltwise_y_in_tensor =
        scope->FindVar(eltwise_y_in->Name())->GetMutable<LoDTensor>();

    // Get batch norm bias
    auto* bn_bias_tensor =
        scope->FindVar(bn_bias->Name())->GetMutable<LoDTensor>();

    // update weights and biases
    float epsilon = boost::get<float>(batch_norm->Op()->GetAttr("epsilon"));
    recompute_bias_and_weights(scope, conv_weight, *bn_scale, *bn_bias_tensor,
                               *bn_mean, *bn_variance, eltwise_y_in_tensor,
                               epsilon);

    // Update the elementwise_add node
    eltwise->Op()->SetAttr("axis", 1);
    eltwise->Op()->SetOutput("Out", std::vector<std::string>({bn_out->Name()}));

    GraphSafeRemoveNodes(
        graph.get(),
        {bn_scale, bn_bias, bn_mean, bn_variance, batch_norm, bn_mean_out,
         bn_variance_out, bn_saved_mean, bn_saved_variance, eltwise_out});

    PADDLE_ENFORCE(subgraph.count(conv_input));
    IR_NODE_LINK_TO(eltwise, bn_out);

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
