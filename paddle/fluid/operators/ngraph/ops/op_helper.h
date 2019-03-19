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
#include "paddle/fluid/operators/ngraph/ops/elementwise_scalar_op.h"
#include "paddle/fluid/platform/ngraph_helper.h"

namespace paddle {
namespace operators {
namespace ngraphs {
std::shared_ptr<ngraph::Node> GetSoftmax(std::shared_ptr<ngraph::Node> x) {
  auto x_shape = x->get_shape();
  int rank = x_shape.size();
  auto x_2d_shape = paddle::platform::FlattenTo2d(x_shape, rank - 1);
  x = paddle::platform::NgReshaper(x, x_2d_shape);

  auto x_max = std::make_shared<ngraph::op::Max>(x, ngraph::AxisSet{1});
  auto x_max_bcast = std::make_shared<ngraph::op::Broadcast>(
      x_max, x_shape, ngraph::AxisSet{1});
  auto x_shifted = x - x_max_bcast;
  auto x_clipped =
      paddle::operators::ngraphs::ElementwiseScalar<ngraph::op::Maximum>(
          -64., x_shifted);
  auto softmax =
      std::make_shared<ngraph::op::Softmax>(x_clipped, ngraph::AxisSet{1});
  return softmax;
}

std::shared_ptr<ngraph::Node> GetCrossEntropy(
    std::shared_ptr<ngraph::Node> x, std::shared_ptr<ngraph::Node> label,
    const bool is_soft_label, int ignore_index) {
  auto label_shape = label->get_shape();
  auto x_shape = x->get_shape();
  auto label_rank = label_shape.size();
  auto x_rank = x_shape.size();
  std::shared_ptr<ngraph::Node> x_2d = x, label_2d = label;
  auto label_2d_shape = label_shape, x_2d_shape = x_shape;

  if (label_rank > 2) {
    label_2d_shape = paddle::platform::FlattenTo2d(label_shape, label_rank - 1);
    label_2d = paddle::platform::NgReshaper(label, label_2d_shape);
  }
  if (x_rank > 2) {
    x_2d_shape = platform::FlattenTo2d(x_shape, x_rank - 1);
    x_2d = platform::NgReshaper(x, x_2d_shape);
  }

  auto batch_size = x_2d_shape.at(0);

  std::shared_ptr<ngraph::Node> node_1_hot = label_2d;
  if (!is_soft_label) {
    auto label_1d =
        platform::NgReshaper(label_2d, ngraph::Shape{label_2d_shape.at(0)});
    node_1_hot = std::make_shared<ngraph::op::OneHot>(label_1d, x_2d_shape, 1);
  }
  if (x->get_element_type() != node_1_hot->get_element_type()) {
    node_1_hot = std::make_shared<ngraph::op::Convert>(node_1_hot,
                                                       x->get_element_type());
  }

  auto node_log = std::make_shared<ngraph::op::Log>(x_2d);
  auto high_clip = ngraph::op::Constant::create(node_log->get_element_type(),
                                                node_log->get_shape(), {1e20});
  auto low_clip = ngraph::op::Constant::create(node_log->get_element_type(),
                                               node_log->get_shape(), {-1e20});
  auto node_min = std::make_shared<ngraph::op::Minimum>(node_log, high_clip);
  auto node_max = std::make_shared<ngraph::op::Maximum>(node_min, low_clip);
  auto node_mul = node_1_hot * node_log;
  auto node_sum =
      std::make_shared<ngraph::op::Sum>(node_mul, ngraph::AxisSet{1});
  auto node_neg = std::make_shared<ngraph::op::Negative>(node_sum);
  auto xe = platform::NgReshaper(node_neg, ngraph::Shape{batch_size, 1});

  if (!is_soft_label) {
    auto ignore_node = ngraph::op::Constant::create(
        label->get_element_type(), label_2d_shape, {ignore_index});
    auto not_equal_node =
        std::make_shared<ngraph::op::NotEqual>(label_2d, ignore_node);
    auto mask = std::make_shared<ngraph::op::Convert>(not_equal_node,
                                                      xe->get_element_type());
    xe = xe * mask;
  }
  return xe;
}
std::shared_ptr<ngraph::Node> GetSoftmaxGrad(
    std::shared_ptr<ngraph::Node> out, std::shared_ptr<ngraph::Node> dout) {
  auto out_shape = out->get_shape();
  int rank = out_shape.size();
  auto out_2d_shape = paddle::platform::FlattenTo2d(out_shape, rank - 1);
  auto dout_2d_shape =
      paddle::platform::FlattenTo2d(dout->get_shape(), rank - 1);
  out = paddle::platform::NgReshaper(out, out_2d_shape);
  dout = paddle::platform::NgReshaper(dout, dout_2d_shape);

  auto node_sum =
      std::make_shared<ngraph::op::Sum>(out * dout, ngraph::AxisSet{1});
  auto node_bcast = std::make_shared<ngraph::op::Broadcast>(
      node_sum, out_2d_shape, ngraph::AxisSet{1});
  auto dx = (dout - node_bcast) * out;
  return dx;
}
std::shared_ptr<ngraph::Node> GetCrossEntropyGrad(
    std::shared_ptr<ngraph::Node> x, std::shared_ptr<ngraph::Node> label,
    std::shared_ptr<ngraph::Node> dy, const bool is_soft_label,
    int ignore_index) {
  auto x_shape = x->get_shape();
  auto rank = x_shape.size();

  std::shared_ptr<ngraph::Node> mask;
  if (!is_soft_label) {
    auto label_shape = label->get_shape();
    label_shape.pop_back();
    label = platform::NgReshaper(label, label_shape);

    auto ignore_node = ngraph::op::Constant::create(
        label->get_element_type(), label_shape, {ignore_index});
    auto not_equal_node =
        std::make_shared<ngraph::op::NotEqual>(label, ignore_node);
    mask = std::make_shared<ngraph::op::Convert>(not_equal_node,
                                                 x->get_element_type());
    mask = std::make_shared<ngraph::op::Broadcast>(mask, x_shape,
                                                   ngraph::AxisSet{rank - 1});

    label = std::make_shared<ngraph::op::OneHot>(label, x_shape, rank - 1);
  }

  auto dy_shape = dy->get_shape();
  dy_shape.pop_back();
  auto dy_reshape = platform::NgReshaper(dy, dy_shape);
  auto dy_bcast = std::make_shared<ngraph::op::Broadcast>(
      dy_reshape, x_shape, ngraph::AxisSet{rank - 1});
  if (x->get_element_type() != label->get_element_type()) {
    label = std::make_shared<ngraph::op::Convert>(label, x->get_element_type());
  }

  auto xe_grad = -label * dy_bcast / x;

  if (!is_soft_label) {
    xe_grad = xe_grad * mask;
  }
  return xe_grad;
}
}  // namespace ngraphs
}  // namespace operators
}  // namespace paddle
