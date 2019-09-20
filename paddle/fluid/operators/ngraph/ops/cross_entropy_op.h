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

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "ngraph/ngraph.hpp"
#include "paddle/fluid/operators/ngraph/ops/op_bridge.h"
#include "paddle/fluid/platform/ngraph_helper.h"

namespace paddle {
namespace operators {
namespace ngraphs {
std::shared_ptr<ngraph::Node> remove_trailing_one(
    const std::shared_ptr<ngraph::Node>& input) {
  auto shape = input->get_shape();
  if (shape.back() == 1 && shape.size() > 1) {
    shape.pop_back();
    return platform::NgReshaper(input, shape);
  } else {
    return input;
  }
}

std::shared_ptr<ngraph::Node> flatten_node(
    const std::shared_ptr<ngraph::Node>& input) {
  auto shape = input->get_shape();
  auto rank = shape.size();
  auto output = input;
  if (rank > 2) {
    auto shape_2d = paddle::platform::FlattenTo2d(shape, rank - 1);
    output = paddle::platform::NgReshaper(input, shape_2d);
  }
  return output;
}

std::shared_ptr<ngraph::Node> convert_to_node_type(
    const std::shared_ptr<ngraph::Node>& input,
    const std::shared_ptr<ngraph::Node>& ref) {
  auto output = input;
  if (input->get_element_type() != ref->get_element_type()) {
    output =
        std::make_shared<ngraph::op::Convert>(input, ref->get_element_type());
  }
  return output;
}

std::shared_ptr<ngraph::Node> create_xe(
    const std::shared_ptr<ngraph::Node>& one_hot,
    const std::shared_ptr<ngraph::Node>& x) {
  auto node_log = std::make_shared<ngraph::op::Log>(x);

  auto node_mul = one_hot * node_log;
  auto node_sum = std::make_shared<ngraph::op::Sum>(
      node_mul, ngraph::AxisSet{x->get_shape().size() - 1});

  auto shape = x->get_shape();
  shape.back() = 1;
  return platform::NgReshaper(-node_sum, shape);
}

std::shared_ptr<ngraph::Node> create_mask(
    const std::shared_ptr<ngraph::Node>& label, int ignore_index) {
  auto ignore_node = paddle::platform::CreateConstant(
      label->get_element_type(), label->get_shape(), {ignore_index});
  auto not_equal_node =
      std::make_shared<ngraph::op::NotEqual>(label, ignore_node);
  return not_equal_node;
}

std::shared_ptr<ngraph::Node> create_one_hot(
    const std::shared_ptr<ngraph::Node>& label,
    const std::shared_ptr<ngraph::Node>& x) {
  auto label_shape = label->get_shape();
  return std::make_shared<ngraph::op::OneHot>(
      remove_trailing_one(label), x->get_shape(), x->get_shape().size() - 1);
}

std::shared_ptr<ngraph::Node> GetCrossEntropy(
    std::shared_ptr<ngraph::Node> x, std::shared_ptr<ngraph::Node> label,
    const bool is_soft_label, int ignore_index) {
  std::shared_ptr<ngraph::Node> node_1_hot = label;
  if (!is_soft_label) {
    node_1_hot = create_one_hot(label, x);
  }
  node_1_hot = convert_to_node_type(node_1_hot, x);

  auto xe = create_xe(node_1_hot, x);
  if (!is_soft_label) {
    auto mask = convert_to_node_type(create_mask(label, ignore_index), xe);
    xe = xe * mask;
  }
  return xe;
}

std::shared_ptr<ngraph::Node> GetCrossEntropyGrad(
    std::shared_ptr<ngraph::Node> x, std::shared_ptr<ngraph::Node> label,
    std::shared_ptr<ngraph::Node> dy, const bool is_soft_label,
    int ignore_index) {
  auto x_shape = x->get_shape();
  auto rank = x_shape.size();

  std::shared_ptr<ngraph::Node> mask;
  if (!is_soft_label) {
    mask = convert_to_node_type(create_mask(label, ignore_index), x);
    mask = std::make_shared<ngraph::op::Broadcast>(
        remove_trailing_one(mask), x_shape, ngraph::AxisSet{rank - 1});
    label = create_one_hot(label, x);
  }

  auto dy_reshape = remove_trailing_one(dy);
  auto dy_bcast = std::make_shared<ngraph::op::Broadcast>(
      dy_reshape, x_shape, ngraph::AxisSet{rank - 1});

  label = convert_to_node_type(label, x);

  auto xe_grad = -label * dy_bcast / x;

  if (!is_soft_label) {
    xe_grad = xe_grad * mask;
  }
  return xe_grad;
}

void BuildCrossEntropyNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto x = paddle::platform::GetInputNode(op, "X", ngb_node_map);
  auto label = paddle::platform::GetInputNode(op, "Label", ngb_node_map);
  auto op_attrs = paddle::framework::AttrReader(op->Attrs());
  const bool is_soft_label = op_attrs.Get<bool>("soft_label");
  int ignore_index = op_attrs.Get<int>("ignore_index");
  auto xe = GetCrossEntropy(x, label, is_soft_label, ignore_index);
  paddle::platform::SetOutputNode(op, "Y", xe, ngb_node_map);
}

void BuildCrossEntropyGradNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto op_attrs = paddle::framework::AttrReader(op->Attrs());
  const bool is_soft_label = op_attrs.Get<bool>("soft_label");
  int ignore_index = op_attrs.Get<int>("ignore_index");
  auto x = paddle::platform::GetInputNode(op, "X", ngb_node_map);
  auto label = paddle::platform::GetInputNode(op, "Label", ngb_node_map);
  auto dy = paddle::platform::GetInputNode(op, "Y@GRAD", ngb_node_map);
  auto xe_grad = GetCrossEntropyGrad(x, label, dy, is_soft_label, ignore_index);
  paddle::platform::SetOutputNode(op, "X@GRAD", xe_grad, ngb_node_map);
}

void BuildCrossEntropy2Node(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto x = paddle::platform::GetInputNode(op, "X", ngb_node_map);
  auto label = paddle::platform::GetInputNode(op, "Label", ngb_node_map);
  auto op_attrs = paddle::framework::AttrReader(op->Attrs());
  int ignore_index = op_attrs.Get<int>("ignore_index");

  auto rank = x->get_shape().size();

  auto one_hot = convert_to_node_type(create_one_hot(label, x), x);
  auto xe = create_xe(one_hot, x);
  auto mask = convert_to_node_type(create_mask(label, ignore_index), xe);

  xe = xe * mask;

  std::shared_ptr<ngraph::Node> node_sum =
      std::make_shared<ngraph::op::Sum>(one_hot * x, ngraph::AxisSet{rank - 1});
  node_sum = paddle::platform::NgReshaper(node_sum, mask->get_shape());
  auto matchx = mask * node_sum;

  paddle::platform::SetOutputNode(op, "MatchX", matchx, ngb_node_map);
  platform::SetOutputNode(op, "XShape", x, ngb_node_map);
  paddle::platform::SetOutputNode(op, "Y", xe, ngb_node_map);
}

void BuildCrossEntropyGrad2Node(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto op_attrs = paddle::framework::AttrReader(op->Attrs());
  int ignore_index = op_attrs.Get<int>("ignore_index");
  auto matchx = paddle::platform::GetInputNode(op, "MatchX", ngb_node_map);
  auto label = paddle::platform::GetInputNode(op, "Label", ngb_node_map);
  auto x = paddle::platform::GetInputNode(op, "XShape", ngb_node_map);
  auto dy = paddle::platform::GetInputNode(op, framework::GradVarName("Y"),
                                           ngb_node_map);

  matchx = remove_trailing_one(matchx);
  label = remove_trailing_one(label);
  x = remove_trailing_one(x);
  dy = remove_trailing_one(dy);

  auto x_shape = x->get_shape();
  auto rank = x_shape.size();

  auto one_hot = convert_to_node_type(create_one_hot(label, x), x);
  auto mask = convert_to_node_type(create_mask(label, ignore_index), x);

  auto zero = paddle::platform::CreateConstant(matchx->get_element_type(),
                                               matchx->get_shape(), {0});
  auto one = paddle::platform::CreateConstant(matchx->get_element_type(),
                                              matchx->get_shape(), {1});
  auto is_zero = std::make_shared<ngraph::op::Equal>(matchx, zero);
  matchx = std::make_shared<ngraph::op::Select>(is_zero, one, matchx);

  auto dy_bcast = std::make_shared<ngraph::op::Broadcast>(
      mask * dy, x_shape, ngraph::AxisSet{rank - 1});
  auto matchx_bcast = std::make_shared<ngraph::op::Broadcast>(
      matchx, x_shape, ngraph::AxisSet{rank - 1});

  auto xe_grad = -dy_bcast * one_hot / matchx_bcast;
  paddle::platform::SetOutputNode(op, framework::GradVarName("X"), xe_grad,
                                  ngb_node_map);
}
}  // namespace ngraphs
}  // namespace operators
}  // namespace paddle

REGISTER_NG_OP(cross_entropy, BuildCrossEntropyNode);
REGISTER_NG_OP(cross_entropy_grad, BuildCrossEntropyGradNode);
REGISTER_NG_OP(cross_entropy2, BuildCrossEntropy2Node);
REGISTER_NG_OP(cross_entropy_grad2, BuildCrossEntropyGrad2Node);
