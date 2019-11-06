/*Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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

std::shared_ptr<ngraph::Node> reshape_reduction(
    std::shared_ptr<ngraph::Node> node, const ngraph::Shape shape,
    int begin_norm_axis) {
  ngraph::Shape keepdims_shape(shape.begin(), shape.begin() + begin_norm_axis);
  return paddle::platform::NgReshaper(node, keepdims_shape);
}

std::shared_ptr<ngraph::Node> broadcast_reduction(
    std::shared_ptr<ngraph::Node> node, const ngraph::Shape shape,
    int begin_norm_axis) {
  ngraph::AxisSet axis_set;
  for (size_t i = begin_norm_axis; i < shape.size(); ++i) axis_set.insert(i);
  auto reshape = reshape_reduction(node, shape, begin_norm_axis);
  return std::make_shared<ngraph::op::Broadcast>(reshape, shape, axis_set);
}

std::shared_ptr<ngraph::Node> reshape_bias_scale(
    std::shared_ptr<ngraph::Node> node, const ngraph::Shape shape,
    int begin_norm_axis) {
  ngraph::Shape keepdims_shape(shape.begin() + begin_norm_axis, shape.end());
  return paddle::platform::NgReshaper(node, keepdims_shape);
}

std::shared_ptr<ngraph::Node> broadcast_bias_scale(
    std::shared_ptr<ngraph::Node> node, const ngraph::Shape shape,
    int begin_norm_axis) {
  auto reshape = reshape_bias_scale(node, shape, begin_norm_axis);
  ngraph::AxisSet axis_set;
  for (int i = 0; i < begin_norm_axis; ++i) axis_set.insert(i);
  return std::make_shared<ngraph::op::Broadcast>(reshape, shape, axis_set);
}

std::shared_ptr<ngraph::Node> flatten(const std::shared_ptr<ngraph::Node>& node,
                                      bool insert_leading_one = false) {
  size_t out = 1;
  for (auto s : node->get_shape()) out *= s;
  if (insert_leading_one) {
    return paddle::platform::NgReshaper(node, ngraph::Shape{1, out});
  } else {
    return paddle::platform::NgReshaper(node, ngraph::Shape{out});
  }
}

static void BuildLayerNormNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto op_attrs = paddle::framework::AttrReader(op->Attrs());
  const auto begin_norm_axis = op_attrs.Get<int>("begin_norm_axis");
  const auto epsilon = op_attrs.Get<float>("epsilon");

  auto x = paddle::platform::GetInputNode(op, "X", ngb_node_map);
  auto scale = paddle::platform::GetInputNode(op, "Scale", ngb_node_map);
  auto bias = paddle::platform::GetInputNode(op, "Bias", ngb_node_map);

  auto shape = x->get_shape();
  std::vector<size_t> reduction_axes(shape.size() - begin_norm_axis);
  std::iota(reduction_axes.begin(), reduction_axes.end(), begin_norm_axis);

  auto mean = ngraph::builder::mean(x, reduction_axes);
  auto broadcast_mean = broadcast_reduction(mean, shape, begin_norm_axis);

  auto delta = x - broadcast_mean;
  auto variance = ngraph::builder::mean(delta * delta, reduction_axes);

  auto eps = paddle::platform::CreateConstant(variance->get_element_type(),
                                              variance->get_shape(), {epsilon});

  auto stddev = std::make_shared<ngraph::op::Sqrt>(variance + eps);
  auto broadcast_stddev = broadcast_reduction(stddev, shape, begin_norm_axis);

  auto norm = delta / broadcast_stddev;

  if (scale) {
    auto broadcast_scale = broadcast_bias_scale(scale, shape, begin_norm_axis);
    norm = norm * broadcast_scale;
  }
  if (bias) {
    auto broadcast_bias = broadcast_bias_scale(bias, shape, begin_norm_axis);
    norm = norm + broadcast_bias;
  }
  mean = flatten(mean);
  variance = flatten(variance);
  paddle::platform::SetOutputNode(op, "Y", norm, ngb_node_map);
  paddle::platform::SetOutputNode(op, "Mean", mean, ngb_node_map);
  paddle::platform::SetOutputNode(op, "Variance", variance, ngb_node_map);
}

static void BuildLayerNormGradNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto op_attrs = paddle::framework::AttrReader(op->Attrs());
  const auto begin_norm_axis = op_attrs.Get<int>("begin_norm_axis");
  const auto epsilon = op_attrs.Get<float>("epsilon");

  auto x = paddle::platform::GetInputNode(op, "X", ngb_node_map);
  auto mean = paddle::platform::GetInputNode(op, "Mean", ngb_node_map);
  auto variance = paddle::platform::GetInputNode(op, "Variance", ngb_node_map);
  auto scale = paddle::platform::GetInputNode(op, "Scale", ngb_node_map);
  auto dy = paddle::platform::GetInputNode(op, framework::GradVarName("Y"),
                                           ngb_node_map);

  auto dx = paddle::platform::GetOutputNode(op, framework::GradVarName("X"),
                                            ngb_node_map);
  auto dscale = paddle::platform::GetOutputNode(
      op, framework::GradVarName("Scale"), ngb_node_map);
  auto dbias = paddle::platform::GetOutputNode(
      op, framework::GradVarName("Bias"), ngb_node_map);

  auto shape = x->get_shape();

  auto broadcast_mean = broadcast_reduction(mean, shape, begin_norm_axis);

  auto delta = x - broadcast_mean;
  auto eps = paddle::platform::CreateConstant(variance->get_element_type(),
                                              variance->get_shape(), {epsilon});

  auto stddev = std::make_shared<ngraph::op::Sqrt>(variance + eps);
  auto broadcast_stddev = broadcast_reduction(stddev, shape, begin_norm_axis);

  auto norm = delta / broadcast_stddev;

  if (dbias) {
    std::vector<size_t> reduction_axes(begin_norm_axis);
    std::iota(reduction_axes.begin(), reduction_axes.end(), 0);
    auto sum_dy = std::make_shared<ngraph::op::Sum>(dy, reduction_axes);
    paddle::platform::SetOutputNode(op, framework::GradVarName("Bias"),
                                    flatten(sum_dy), ngb_node_map);
  }
  if (dscale) {
    std::vector<size_t> reduction_axes(begin_norm_axis);
    std::iota(reduction_axes.begin(), reduction_axes.end(), 0);
    auto sum_dy = std::make_shared<ngraph::op::Sum>(dy * norm, reduction_axes);
    paddle::platform::SetOutputNode(op, framework::GradVarName("Scale"),
                                    flatten(sum_dy), ngb_node_map);
  }

  if (dx) {
    std::shared_ptr<ngraph::Node> dx_end = dy / broadcast_stddev;
    if (dscale)
      dx_end = dx_end * broadcast_bias_scale(scale, shape, begin_norm_axis);

    std::vector<size_t> reduction_axes(shape.size() - begin_norm_axis);
    std::iota(reduction_axes.begin(), reduction_axes.end(), begin_norm_axis);

    auto dx_mean = broadcast_reduction(
        ngraph::builder::mean(-dx_end, reduction_axes), shape, begin_norm_axis);

    auto dx_std =
        norm * broadcast_reduction(
                   ngraph::builder::mean(-dx_end * norm, reduction_axes), shape,
                   begin_norm_axis);

    paddle::platform::SetOutputNode(op, framework::GradVarName("X"),
                                    dx_end + dx_mean + dx_std, ngb_node_map);
  }
}

REGISTER_NG_OP(layer_norm, BuildLayerNormNode);
REGISTER_NG_OP(layer_norm_grad, BuildLayerNormGradNode);

}  // namespace ngraphs
}  // namespace operators
}  // namespace paddle
