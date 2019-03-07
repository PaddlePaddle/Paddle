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
#include "paddle/fluid/operators/ngraph/ops/elementwise_scalar_op.h"
#include "paddle/fluid/operators/ngraph/ops/op_bridge.h"
#include "paddle/fluid/platform/ngraph_helper.h"

namespace paddle {
namespace operators {
namespace ngraphs {

void BuildMeanNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto input = paddle::platform::GetInputNode(op, "X", ngb_node_map);
  ngraph::AxisSet axes;
  for (size_t i = 0; i < input->get_shape().size(); ++i) {
    axes.insert(i);
  }

  auto mean = ngraph::builder::mean(input, axes);
  auto mean_1d = std::make_shared<ngraph::op::Reshape>(
      mean, ngraph::AxisVector{}, ngraph::Shape{1});
  paddle::platform::SetOutputNode(op, "Out", mean_1d, ngb_node_map);
}

void BuildMeanGradNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto x = paddle::platform::GetInputNode(op, "X", ngb_node_map);
  auto og = paddle::platform::GetInputNode(op, "Out@GRAD", ngb_node_map);
  auto x_shape = x->get_shape();
  float x_size = std::accumulate(std::begin(x_shape), std::end(x_shape), 1,
                                 std::multiplies<float>());
  auto node_const = ngraph::op::Constant::create(og->get_element_type(),
                                                 ngraph::Shape{1}, {x_size});
  auto node_div = std::make_shared<ngraph::op::Divide>(og, node_const);

  auto result = ElementwiseScalar<ngraph::op::Add>(
      og / node_const,
      ngraph::op::Constant::create(og->get_element_type(), x_shape, {0}));
  paddle::platform::SetOutputNode(op, "X@GRAD", result, ngb_node_map);
}
}  // namespace ngraphs
}  // namespace operators
}  // namespace paddle

REGISTER_NG_OP(mean, BuildMeanNode);
REGISTER_NG_OP(mean_grad, BuildMeanGradNode);
