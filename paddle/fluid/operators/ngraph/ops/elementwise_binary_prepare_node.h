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

#include <string>
#include <vector>

#include "ngraph/ngraph.hpp"
#include "paddle/fluid/platform/ngraph_helper.h"

namespace paddle {
namespace operators {
namespace ngraphs {

ngraph::NodeVector ElementwiseBinaryNodePrepare(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto op_attrs = paddle::framework::AttrReader(op->Attrs());
  int axis = op_attrs.Get<int>("axis");
  auto lhs = paddle::platform::GetInputNode(op, "X", ngb_node_map);
  auto rhs = paddle::platform::GetInputNode(op, "Y", ngb_node_map);

  auto lhs_shape = lhs->get_shape();
  auto rhs_shape = rhs->get_shape();

  PADDLE_ENFORCE_GE(lhs_shape.size(), rhs_shape.size(),
                    "Rank of first input must >= rank of second input.");
  if (lhs_shape == rhs_shape) {
    return ngraph::NodeVector{lhs, rhs};
  }
  axis = (axis == -1 ? lhs_shape.size() - rhs_shape.size() : axis);
  PADDLE_ENFORCE(axis >= 0 && axis < (int)(lhs_shape.size()),
                 "Axis should be in range [0, lhs_shape)");
  paddle::platform::TrimTrailingSingularDims(&rhs_shape);
  axis = (rhs_shape.size() == 0) ? lhs_shape.size() : axis;

  int pre, n, post;
  paddle::platform::GetMidDims(lhs_shape, rhs_shape, axis, &pre, &n, &post);

  ngraph::Shape l_shape{};
  l_shape.push_back(pre);
  l_shape.push_back(n);
  l_shape.push_back(post);

  std::vector<size_t> rhs_order(rhs->get_shape().size());
  std::iota(std::begin(rhs_order), std::end(rhs_order), 0);
  ngraph::Shape r_shape{};
  r_shape.push_back(n);
  auto rhs_reshape = std::make_shared<ngraph::op::Reshape>(
      rhs, ngraph::AxisVector(rhs_order), r_shape);
  auto rhs_bcast = std::make_shared<ngraph::op::Broadcast>(
      rhs_reshape, l_shape, ngraph::AxisSet{0, 2});
  std::vector<size_t> bcast_order(rhs_bcast->get_shape().size());
  std::iota(std::begin(bcast_order), std::end(bcast_order), 0);
  std::shared_ptr<ngraph::Node> rhs_bcast_reshape =
      std::make_shared<ngraph::op::Reshape>(
          rhs_bcast, ngraph::AxisVector(bcast_order), lhs_shape);
  return ngraph::NodeVector{lhs, rhs_bcast_reshape};
}
}  // namespace ngraphs
}  // namespace operators
}  // namespace paddle
