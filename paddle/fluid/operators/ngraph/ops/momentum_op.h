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
#include "paddle/fluid/operators/ngraph/ops/op_bridge.h"
#include "paddle/fluid/platform/ngraph_helper.h"

namespace paddle {
namespace operators {
namespace ngraphs {

void BuildMomentumNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto op_attrs = paddle::framework::AttrReader(op->Attrs());
  auto param = paddle::platform::GetInputNode(op, "Param", ngb_node_map);
  auto grad = paddle::platform::GetInputNode(op, "Grad", ngb_node_map);
  auto velocity = paddle::platform::GetInputNode(op, "Velocity", ngb_node_map);
  auto learning_rate =
      paddle::platform::GetInputNode(op, "LearningRate", ngb_node_map);

  auto mu = op_attrs.Get<float>("mu");
  bool use_nesterov = op_attrs.Get<bool>("use_nesterov");

  auto param_shape = param->get_shape();
  auto velocity_shape = velocity->get_shape();
  auto grad_shape = grad->get_shape();
  auto lr_shape = learning_rate->get_shape();

  auto shape_velocity = ngraph::Shape{velocity_shape};
  auto mu_create =
      ngraph::op::Constant::create(ngraph::element::f32, shape_velocity, {mu});

  auto vel_mul = std::make_shared<ngraph::op::Multiply>(velocity, mu_create);
  auto vel_out = std::make_shared<ngraph::op::Add>(vel_mul, grad);

  ngraph::NodeVector result;
  if (use_nesterov) {
    auto mul_res = std::make_shared<ngraph::op::Multiply>(vel_out, mu_create);
    auto add_res = std::make_shared<ngraph::op::Add>(grad, mul_res);

    auto add_2d = paddle::platform::FlattenTo2d(add_res->get_shape(), 0);
    auto vel_reshape = paddle::platform::NgReshaper(vel_out, add_2d);

    auto lr_bcast = std::make_shared<ngraph::op::Broadcast>(
        learning_rate, vel_reshape->get_shape(),
        ngraph::AxisSet{vel_reshape->get_shape().size() - 1});

    auto lr_1d = paddle::platform::FlattenTo1d(lr_bcast->get_shape(), 0);
    auto lr_reshape = std::make_shared<ngraph::op::Reshape>(
        lr_bcast, ngraph::AxisVector{0, 1}, lr_1d);

    lr_reshape = std::make_shared<ngraph::op::Reshape>(
        lr_reshape, ngraph::AxisVector{0}, param->get_shape());

    auto mul_res1 = std::make_shared<ngraph::op::Multiply>(add_res, lr_reshape);
    auto res = std::make_shared<ngraph::op::Subtract>(param, mul_res1);
    paddle::platform::SetOutputNode(op, "ParamOut", res, ngb_node_map);
  } else {
    auto vel_2d = paddle::platform::FlattenTo2d(vel_out->get_shape(), 0);
    auto vel_reshape = paddle::platform::NgReshaper(vel_out, vel_2d);

    auto lr_bcast = std::make_shared<ngraph::op::Broadcast>(
        learning_rate, vel_reshape->get_shape(),
        ngraph::AxisSet{vel_reshape->get_shape().size() - 1});

    auto lr_1d = paddle::platform::FlattenTo1d(lr_bcast->get_shape(), 0);
    auto lr_reshape = std::make_shared<ngraph::op::Reshape>(
        lr_bcast, ngraph::AxisVector{0, 1}, lr_1d);

    lr_reshape = std::make_shared<ngraph::op::Reshape>(
        lr_reshape, ngraph::AxisVector{0}, param->get_shape());

    auto mul_result =
        std::make_shared<ngraph::op::Multiply>(lr_reshape, vel_out);

    auto res = std::make_shared<ngraph::op::Subtract>(param, mul_result);
    paddle::platform::SetOutputNode(op, "ParamOut", res, ngb_node_map);
  }
  paddle::platform::SetOutputNode(op, "VelocityOut", vel_out, ngb_node_map);
}

}  // namespace ngraphs
}  // namespace operators
}  // namespace paddle

REGISTER_NG_OP(momentum, BuildMomentumNode);
