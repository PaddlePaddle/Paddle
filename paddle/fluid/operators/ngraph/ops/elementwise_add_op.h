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

void BuildElementwiseAddNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  BuildElementwiseBinaryNode<ngraph::op::Add>(op, ngb_node_map);
}

void BuildElementwiseAddGradNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto op_attrs = paddle::framework::AttrReader(op->Attrs());
  int axis = op_attrs.Get<int>("axis");

  auto dout = paddle::platform::GetInputNode(op, "Out@GRAD", ngb_node_map);
  auto y = paddle::platform::GetInputNode(op, "Y", ngb_node_map);
  auto dout_shape = dout->get_shape();
  auto y_shape = y->get_shape();

  if (dout_shape == y_shape) {
    paddle::platform::SetOutputNode(op, "X@GRAD", dout, ngb_node_map);
    paddle::platform::SetOutputNode(op, "Y@GRAD", dout, ngb_node_map);
  } else {
    axis = (axis == -1 ? dout_shape.size() - y_shape.size() : axis);
    paddle::platform::TrimTrailingSingularDims(&y_shape);
    axis = (y_shape.size() == 0 ? dout_shape.size() : axis);

    int pre, n, post;
    paddle::platform::GetMidDims(dout_shape, y_shape, axis, &pre, &n, &post);

    ngraph::Shape lhs_shape{};
    lhs_shape.push_back(pre);
    lhs_shape.push_back(n);
    if (post != 1) {
      lhs_shape.push_back(post);
    }

    std::vector<size_t> lhs_order(dout_shape.size());
    std::iota(std::begin(lhs_order), std::end(lhs_order), 0);
    auto dout_reshape = std::make_shared<ngraph::op::Reshape>(
        dout, ngraph::AxisVector(lhs_order), lhs_shape);

    ngraph::AxisSet axis_set{0};
    if (post != 1) {
      axis_set.insert(2);
    }

    auto dout_sum = std::make_shared<ngraph::op::Sum>(dout_reshape, axis_set);
    auto dy = std::make_shared<ngraph::op::Reshape>(
        dout_sum, ngraph::AxisVector{0}, y->get_shape());

    paddle::platform::SetOutputNode(op, "X@GRAD", dout, ngb_node_map);
    paddle::platform::SetOutputNode(op, "Y@GRAD", dy, ngb_node_map);
  }
}
}  // namespace ngraphs
}  // namespace operators
}  // namespace paddle

REGISTER_NG_OP(elementwise_add, BuildElementwiseAddNode);
REGISTER_NG_OP(elementwise_add_grad, BuildElementwiseAddGradNode);
