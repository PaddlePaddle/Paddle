/*Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "ngraph/ngraph.hpp"
#include "paddle/fluid/operators/ngraph/ops/elementwise_node.h"
#include "paddle/fluid/operators/ngraph/ops/elementwise_scalar_op.h"
#include "paddle/fluid/operators/ngraph/ops/op_bridge.h"
#include "paddle/fluid/platform/ngraph_helper.h"

namespace paddle {
namespace operators {
namespace ngraphs {

void BuildBatchNormNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto op_attrs = paddle::framework::AttrReader(op->Attrs());
  auto& data_layout = op_attrs.Get<std::string>("data_layout");

  auto bias = paddle::platform::GetInputNode(op, "Bias", ngb_node_map);
  auto mean = paddle::platform::GetInputNode(op, "Mean", ngb_node_map);
  auto variance = paddle::platform::GetInputNode(op, "Variance", ngb_node_map);
  auto scale = paddle::platform::GetInputNode(op, "Scale", ngb_node_map);
  auto x = paddle::platform::GetInputNode(op, "X", ngb_node_map);

  const bool is_test = op_attrs.Get<bool>("is_test");
  const float epsilon = op_attrs.Get<float>("epsilon");
  const float momentum = op_attrs.Get<float>("momentum");

  PADDLE_ENFORCE(
      data_layout == "NHWC" || data_layout == "NCHW" || data_layout == "NC",
      "The BatchNorm operator only supports NHWC/NCHW/NC data format");

  if (data_layout == "NHWC") {
    x = paddle::platform::Nhwc2Nchw(x);
  }

  std::shared_ptr<ngraph::Node> mean_out, saved_mean, saved_variance,
      variance_out, y;

  if (!is_test) {
    auto BN = std::make_shared<ngraph::op::BatchNormTraining>(epsilon, scale,
                                                              bias, x);
    y = std::make_shared<ngraph::op::GetOutputElement>(BN, 0);
    saved_mean = std::make_shared<ngraph::op::GetOutputElement>(BN, 1);
    saved_variance = std::make_shared<ngraph::op::GetOutputElement>(BN, 2);

    mean_out = std::make_shared<ngraph::op::Add>(
        paddle::operators::ngraphs::ElementwiseScalar<ngraph::op::Multiply>(
            momentum, mean),
        paddle::operators::ngraphs::ElementwiseScalar<ngraph::op::Multiply>(
            1. - momentum, saved_mean));
    variance_out = std::make_shared<ngraph::op::Add>(
        paddle::operators::ngraphs::ElementwiseScalar<ngraph::op::Multiply>(
            momentum, variance),
        paddle::operators::ngraphs::ElementwiseScalar<ngraph::op::Multiply>(
            1. - momentum, saved_variance));

    if (data_layout == "NHWC") {
      y = paddle::platform::Nchw2Nhwc(y);
    }

    paddle::platform::SetOutputNode(op, "MeanOut", mean_out, ngb_node_map);
    paddle::platform::SetOutputNode(op, "VarianceOut", variance_out,
                                    ngb_node_map);
    paddle::platform::SetOutputNode(op, "SavedMean", saved_mean, ngb_node_map);
    paddle::platform::SetOutputNode(op, "SavedVariance", saved_variance,
                                    ngb_node_map);
    paddle::platform::SetOutputNode(op, "Y", y, ngb_node_map);
  } else {
    y = std::make_shared<ngraph::op::BatchNormInference>(epsilon, scale, bias,
                                                         x, mean, variance);
    paddle::platform::SetOutputNode(op, "Y", y, ngb_node_map);
  }
}

void BuildBatchNormGradNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto op_attrs = paddle::framework::AttrReader(op->Attrs());
  auto& data_layout = op_attrs.Get<std::string>("data_layout");

  auto bias = paddle::platform::GetInputNode(op, "Bias", ngb_node_map);
  auto saved_mean =
      paddle::platform::GetInputNode(op, "SavedMean", ngb_node_map);
  auto saved_variance =
      paddle::platform::GetInputNode(op, "SavedVariance", ngb_node_map);
  auto scale = paddle::platform::GetInputNode(op, "Scale", ngb_node_map);
  auto x = paddle::platform::GetInputNode(op, "X", ngb_node_map);
  auto dy = paddle::platform::GetInputNode(op, "Y@GRAD", ngb_node_map);
  auto x_shape = x->get_shape();
  auto dy_shape = dy->get_shape();

  PADDLE_ENFORCE(x_shape.size() == 2 || x_shape.size() == 4,
                 "BN grap input size needs to be 2 or 4");
  PADDLE_ENFORCE_EQ(x_shape.size(), dy_shape.size(),
                    "BN grap input and delta size needs to be equal");
  PADDLE_ENFORCE(
      data_layout == "NHWC" || data_layout == "NCHW" || data_layout == "NC",
      "The BatchNorm operator only supports NHWC/NCHW/NC data format");

  if (x_shape.size() == 2) {
    x = std::make_shared<ngraph::op::Reshape>(
        x, ngraph::AxisVector{0, 1},
        ngraph::Shape{x_shape.at(0), x_shape.at(1), 1, 1});
    dy = std::make_shared<ngraph::op::Reshape>(
        dy, ngraph::AxisVector{0, 1},
        ngraph::Shape{dy_shape.at(0), dy_shape.at(1), 1, 1});
  }

  if (data_layout == "NHWC") {
    x = paddle::platform::Nhwc2Nchw(dy);
    dy = paddle::platform::Nhwc2Nchw(dy);
  }
  const float epsilon = op_attrs.Get<float>("epsilon");

  auto bn_bprop = std::make_shared<ngraph::op::BatchNormTrainingBackprop>(
      epsilon, scale, bias, x, saved_mean, saved_variance, dy);

  std::shared_ptr<ngraph::Node> dx =
      std::make_shared<ngraph::op::GetOutputElement>(bn_bprop, 0);
  auto dscale = std::make_shared<ngraph::op::GetOutputElement>(bn_bprop, 1);
  auto dbias = std::make_shared<ngraph::op::GetOutputElement>(bn_bprop, 2);
  paddle::platform::SetOutputNode(op, "Bias@GRAD", dbias, ngb_node_map);
  paddle::platform::SetOutputNode(op, "Scale@GRAD", dscale, ngb_node_map);
  if (x_shape.size() == 2) {
    paddle::platform::SetOutputNode(
        op, "X@GRAD", paddle::platform::NgReshaper(dx, x_shape), ngb_node_map);
  } else {
    if (data_layout == "NHWC") {
      dx = paddle::platform::Nchw2Nhwc(dx);
    }
    paddle::platform::SetOutputNode(op, "X@GRAD", dx, ngb_node_map);
  }
}
}  // namespace ngraphs
}  // namespace operators
}  // namespace paddle

REGISTER_NG_OP(batch_norm, BuildBatchNormNode);
REGISTER_NG_OP(batch_norm_grad, BuildBatchNormGradNode);
