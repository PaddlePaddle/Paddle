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

#include "ngraph/ngraph.hpp"
#include "paddle/fluid/operators/ngraph/ops/op_bridge.h"
#include "paddle/fluid/platform/ngraph_helper.h"

namespace paddle {
namespace operators {
namespace ngraphs {

void BuildGeluNode(
    const std::shared_ptr<framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto input = platform::GetInputNode(op, "X", ngb_node_map);
  auto half = paddle::platform::CreateConstant(input->get_element_type(),
                                               input->get_shape(), {0.5});
  auto one = paddle::platform::CreateConstant(input->get_element_type(),
                                              input->get_shape(), {1});
  auto sqrt_two =
      std::make_shared<ngraph::op::Sqrt>(paddle::platform::CreateConstant(
          input->get_element_type(), input->get_shape(), {2}));
  auto out = half * input *
             (one + std::make_shared<ngraph::op::Erf>(input / sqrt_two));
  platform::SetOutputNode(op, "Out", out, ngb_node_map);
}

void BuildGeluGradNode(
    const std::shared_ptr<framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto input = platform::GetInputNode(op, "X", ngb_node_map);
  auto dout = platform::GetInputNode(op, "Out@GRAD", ngb_node_map);
  auto half = paddle::platform::CreateConstant(input->get_element_type(),
                                               input->get_shape(), {0.5});
  auto minus_half = paddle::platform::CreateConstant(
      input->get_element_type(), input->get_shape(), {-0.5});
  auto one = paddle::platform::CreateConstant(input->get_element_type(),
                                              input->get_shape(), {1});
  auto two = paddle::platform::CreateConstant(input->get_element_type(),
                                              input->get_shape(), {2});
  auto pi = paddle::platform::CreateConstant(
      input->get_element_type(), input->get_shape(), {3.14159265359});
  auto sqrt_two = std::make_shared<ngraph::op::Sqrt>(two);
  auto sqrt_pi = std::make_shared<ngraph::op::Sqrt>(pi);

  auto first =
      half * (one + std::make_shared<ngraph::op::Erf>(input * one / sqrt_two));
  auto second = half * (two / sqrt_pi) * (one / sqrt_two) * input *
                std::make_shared<ngraph::op::Exp>(minus_half * input * input);
  auto gelu_grad = dout * (first + second);
  platform::SetOutputNode(op, "X@GRAD", gelu_grad, ngb_node_map);
}

void BuildReluGradNode(
    const std::shared_ptr<framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto out = platform::GetInputNode(op, "Out", ngb_node_map);
  auto dout = platform::GetInputNode(op, "Out@GRAD", ngb_node_map);
  auto relu_grad = std::make_shared<ngraph::op::ReluBackprop>(out, dout);
  platform::SetOutputNode(op, "X@GRAD", relu_grad, ngb_node_map);
}

void BuildSquareNode(
    const std::shared_ptr<framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto input = platform::GetInputNode(op, "X", ngb_node_map);
  auto out = input * input;
  platform::SetOutputNode(op, "Out", out, ngb_node_map);
}

void BuildTanhGradNode(
    const std::shared_ptr<framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto out = platform::GetInputNode(op, "Out", ngb_node_map);
  auto dout = platform::GetInputNode(op, "Out@GRAD", ngb_node_map);
  auto shape = out->get_shape();
  auto node_const =
      ngraph::op::Constant::create(ngraph::element::f32, shape, {1});
  auto result = dout * (node_const - out * out);
  platform::SetOutputNode(op, "X@GRAD", result, ngb_node_map);
}
}  // namespace ngraphs
}  // namespace operators
}  // namespace paddle

REGISTER_NG_OP(gelu, BuildGeluNode);
REGISTER_NG_OP(gelu_grad, BuildGeluGradNode);
REGISTER_NG_OP(relu_grad, BuildReluGradNode);
REGISTER_NG_OP(square, BuildSquareNode);
REGISTER_NG_OP(tanh_grad, BuildTanhGradNode);
