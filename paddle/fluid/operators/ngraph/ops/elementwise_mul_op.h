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
#include "paddle/fluid/operators/ngraph/ops/op_bridge.h"
#include "paddle/fluid/platform/ngraph_helper.h"

namespace paddle {
namespace operators {
namespace ngraphs {

void BuildElementwiseMulNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  BuildElementwiseBinaryNode<ngraph::op::Multiply>(op, ngb_node_map);
}

void BuildElementwiseMulGradNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto op_attrs = paddle::framework::AttrReader(op->Attrs());
  int axis = op_attrs.Get<int>("axis");

  auto dout = paddle::platform::GetInputNode(op, "Out@GRAD", ngb_node_map);
  auto y = paddle::platform::GetInputNode(op, "Y", ngb_node_map);
  auto x = paddle::platform::GetInputNode(op, "X", ngb_node_map);
  auto dout_shape = dout->get_shape();
  auto y_shape = y->get_shape();
  auto x_shape = x->get_shape();
  if (dout->get_element_type() != y->get_element_type()) {
    y = std::make_shared<ngraph::op::Convert>(y, dout->get_element_type());
  }
  if (dout_shape == y_shape) {
    auto dx = std::make_shared<ngraph::op::Multiply>(dout, y);
    auto dy = std::make_shared<ngraph::op::Multiply>(dout, x);
    paddle::platform::SetOutputNode(op, "X@GRAD", dx, ngb_node_map);
    paddle::platform::SetOutputNode(op, "Y@GRAD", dy, ngb_node_map);
  } else {
    auto dy_hd = std::make_shared<ngraph::op::Multiply>(dout, x);
    auto dy_hd_shape = dy_hd->get_shape();
    axis = (axis == -1 ? dy_hd_shape.size() - y_shape.size() : axis);
    paddle::platform::TrimTrailingSingularDims(&y_shape);
    axis = (y_shape.size() == 0 ? dy_hd_shape.size() : axis);
    int pre, n, post;
    paddle::platform::GetMidDims(dy_hd_shape, y_shape, axis, &pre, &n, &post);
    ngraph::Shape lhs_shape{};
    lhs_shape.push_back(pre);
    lhs_shape.push_back(n);
    if (post != 1) {
      lhs_shape.push_back(post);
    }

    std::vector<size_t> dy_order(dout_shape.size());
    std::iota(std::begin(dy_order), std::end(dy_order), 0);
    auto dy_hd_reshape = std::make_shared<ngraph::op::Reshape>(
        dy_hd, ngraph::AxisVector(dy_order), lhs_shape);

    ngraph::AxisSet axis_set{0};
    if (post != 1) {
      axis_set.insert(2);
    }

    auto dy_sum = std::make_shared<ngraph::op::Sum>(dy_hd_reshape, axis_set);
    auto dy_sum_yshape = std::make_shared<ngraph::op::Reshape>(
        dy_sum, ngraph::AxisVector{0}, y->get_shape());
    paddle::platform::SetOutputNode(op, "Y@GRAD", dy_sum_yshape, ngb_node_map);

    y_shape = y->get_shape();
    std::vector<size_t> y_order(y_shape.size() == 0 ? 1 : y_shape.size());
    std::iota(std::begin(y_order), std::end(y_order), 0);
    auto y_reshape = std::make_shared<ngraph::op::Reshape>(
        y, ngraph::AxisVector(y_order), ngraph::Shape{(size_t)n});
    auto y_broadcast =
        std::make_shared<ngraph::op::Broadcast>(y_reshape, lhs_shape, axis_set);
    std::vector<size_t> lhs_order(lhs_shape.size());
    std::iota(std::begin(lhs_order), std::end(lhs_order), 0);
    auto y_broadcast_reshape = std::make_shared<ngraph::op::Reshape>(
        y_broadcast, ngraph::AxisVector(lhs_order), dout_shape);
    auto dx = std::make_shared<ngraph::op::Multiply>(y_broadcast_reshape, dout);
    paddle::platform::SetOutputNode(op, "X@GRAD", dx, ngb_node_map);
  }
}
}  // namespace ngraphs
}  // namespace operators
}  // namespace paddle

REGISTER_NG_OP(elementwise_mul, BuildElementwiseMulNode);
REGISTER_NG_OP(elementwise_mul_grad, BuildElementwiseMulGradNode);
