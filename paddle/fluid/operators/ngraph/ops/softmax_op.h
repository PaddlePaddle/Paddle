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
#include "paddle/fluid/operators/ngraph/ops/elementwise_scalar_op.h"
#include "paddle/fluid/operators/ngraph/ops/op_bridge.h"
#include "paddle/fluid/platform/ngraph_helper.h"

namespace paddle {
namespace operators {
namespace ngraphs {

std::shared_ptr<ngraph::Node> GetSoftmax(std::shared_ptr<ngraph::Node> x,
                                         int axis = -1) {
  auto x_shape = x->get_shape();
  size_t rank = x_shape.size();
  size_t softmax_axis = axis;
  if (axis < 0) softmax_axis = rank + axis;

  auto x_max =
      std::make_shared<ngraph::op::Max>(x, ngraph::AxisSet{softmax_axis});
  auto x_max_bcast = std::make_shared<ngraph::op::Broadcast>(
      x_max, x_shape, ngraph::AxisSet{softmax_axis});
  auto x_shifted = x - x_max_bcast;
  auto x_clipped =
      paddle::operators::ngraphs::ElementwiseScalar<ngraph::op::Maximum>(
          -64., x_shifted);
  auto softmax = std::make_shared<ngraph::op::Softmax>(
      x_clipped, ngraph::AxisSet{softmax_axis});
  return softmax;
}

std::shared_ptr<ngraph::Node> GetSoftmaxGrad(std::shared_ptr<ngraph::Node> out,
                                             std::shared_ptr<ngraph::Node> dout,
                                             int axis = -1) {
  auto out_shape = out->get_shape();
  size_t rank = out_shape.size();
  size_t softmax_axis = axis;
  if (axis < 0) softmax_axis = rank + axis;

  auto node_sum = std::make_shared<ngraph::op::Sum>(
      out * dout, ngraph::AxisSet{softmax_axis});
  auto node_bcast = std::make_shared<ngraph::op::Broadcast>(
      node_sum, out_shape, ngraph::AxisSet{softmax_axis});
  auto dx = (dout - node_bcast) * out;
  return dx;
}

void BuildSoftmaxNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto op_attrs = framework::AttrReader(op->Attrs());
  auto x = paddle::platform::GetInputNode(op, "X", ngb_node_map);
  auto softmax = GetSoftmax(x, op_attrs.Get<int>("axis"));
  paddle::platform::SetOutputNode(op, "Out", softmax, ngb_node_map);
}

void BuildSoftmaxGradNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto op_attrs = framework::AttrReader(op->Attrs());
  auto out = paddle::platform::GetInputNode(op, "Out", ngb_node_map);
  auto dout = paddle::platform::GetInputNode(op, "Out@GRAD", ngb_node_map);
  auto dx = GetSoftmaxGrad(out, dout, op_attrs.Get<int>("axis"));
  paddle::platform::SetOutputNode(op, "X@GRAD", dx, ngb_node_map);
}
}  // namespace ngraphs
}  // namespace operators
}  // namespace paddle

REGISTER_NG_OP(softmax, BuildSoftmaxNode);
REGISTER_NG_OP(softmax_grad, BuildSoftmaxGradNode);
