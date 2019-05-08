/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include <vector>
#include "ngraph/ngraph.hpp"
#include "paddle/fluid/operators/ngraph/ops/op_bridge.h"
#include "paddle/fluid/platform/ngraph_helper.h"

namespace paddle {
namespace operators {
namespace ngraphs {

void BuildGatherNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto x = platform::GetInputNode(op, "X", ngb_node_map);
  auto index = platform::GetInputNode(op, "Index", ngb_node_map);
  auto x_shape = x->get_shape();
  size_t axis_1 = x_shape[0];
  size_t axis_2 = 1;
  if (x_shape.size() > 1) {
    axis_2 = std::accumulate(std::begin(x_shape) + 1, std::end(x_shape), 1,
                             std::multiplies<size_t>());
  }
  std::vector<size_t> x_order(x_shape.size());
  std::iota(std::begin(x_order), std::end(x_order), 0);
  auto x_reshape = std::make_shared<ngraph::op::Reshape>(
      x, ngraph::AxisVector(x_order), ngraph::Shape{axis_1, axis_2});
  auto x_reshape_shape = x_reshape->get_shape();
  auto result = std::make_shared<ngraph::op::EmbeddingLookup>(index, x_reshape);
  auto result_shape = result->get_shape();
  std::vector<size_t> out_shape(x_shape);
  out_shape[0] = result_shape[0];
  std::vector<size_t> axis_vector;
  for (size_t i = 0; i < result_shape.size(); i++) {
    axis_vector.push_back(i);
  }
  auto out = std::make_shared<ngraph::op::Reshape>(
      result, ngraph::AxisVector(axis_vector), out_shape);
  paddle::platform::SetOutputNode(op, "Out", out, ngb_node_map);
}
}  // namespace ngraphs
}  // namespace operators
}  // namespace paddle

REGISTER_NG_OP(gather, BuildGatherNode);
