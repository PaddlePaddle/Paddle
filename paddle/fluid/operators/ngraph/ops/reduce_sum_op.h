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

#include <algorithm>
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

void BuildReduceSumNode(
    const std::shared_ptr<paddle::framework::OperatorBase> &op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto input = paddle::platform::GetInputNode(op, "X", ngb_node_map);
  auto op_attrs = paddle::framework::AttrReader(op->Attrs());
  bool reduce_all = op_attrs.Get<bool>("reduce_all");
  bool keep_dim = op_attrs.Get<bool>("keep_dim");
  std::vector<int> dim = op_attrs.Get<std::vector<int>>("dim");
  auto input_shape = input->get_shape();
  ngraph::AxisSet axes;
  if (reduce_all == true) {
    for (size_t i = 0; i < input_shape.size(); ++i) {
      axes.insert(i);
    }
  } else {
    for (auto &i : dim) {
      if (i < 0) {
        axes.insert(input_shape.size() + i);
      } else {
        axes.insert(i);
      }
    }
  }
  std::shared_ptr<ngraph::Node> reduce_sum =
      std::make_shared<ngraph::op::Sum>(input, axes);

  if (keep_dim == true) {
    std::vector<size_t> dim_shape;
    std::copy(input_shape.begin(), input_shape.end(),
              std::back_inserter(dim_shape));
    for (auto &i : dim) {
      if (i < 0) {
        i = input_shape.size() + i;
      }
      dim_shape[i] = 1;
    }

    std::vector<size_t> axis_vector(input_shape.size() - dim.size());
    std::iota(axis_vector.begin(), axis_vector.end(), 0);

    auto reduce_sum_dim = std::make_shared<ngraph::op::Reshape>(
        reduce_sum, ngraph::AxisVector(axis_vector), ngraph::Shape(dim_shape));

    paddle::platform::SetOutputNode(op, "Out", reduce_sum_dim, ngb_node_map);
  } else {
    if (reduce_sum->get_shape() == ngraph::Shape{}) {
      reduce_sum = paddle::platform::NgReshaper(reduce_sum, ngraph::Shape{1});
    }
    paddle::platform::SetOutputNode(op, "Out", reduce_sum, ngb_node_map);
  }
}

void BuildReduceSumGradNode(
    const std::shared_ptr<paddle::framework::OperatorBase> &op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto x = paddle::platform::GetInputNode(op, "X", ngb_node_map);
  auto og = paddle::platform::GetInputNode(op, "Out@GRAD", ngb_node_map);
  auto op_attrs = paddle::framework::AttrReader(op->Attrs());
  std::vector<int> dim = op_attrs.Get<std::vector<int>>("dim");
  bool reduce_all = op_attrs.Get<bool>("reduce_all");
  bool keep_dim = op_attrs.Get<bool>("keep_dim");

  auto og_shape = og->get_shape();
  auto x_shape = x->get_shape();
  float x_size = std::accumulate(std::begin(x_shape), std::end(x_shape), 1,
                                 std::multiplies<float>());
  float og_size = std::accumulate(std::begin(og_shape), std::end(og_shape), 1,
                                  std::multiplies<float>());
  ngraph::AxisSet axes;

  if (reduce_all == true) {
    for (size_t i = 0; i < x_shape.size(); i++) {
      axes.insert(i);
    }
  } else {
    for (auto &i : dim) {
      if (i < 0) {
        axes.insert(x_shape.size() + i);
      } else {
        axes.insert(i);
      }
    }
  }
  std::vector<size_t> axis_vector(og_shape.size());
  std::iota(axis_vector.begin(), axis_vector.end(), 0);
  std::vector<size_t> dim_shape;

  for (size_t i = 0; i < x_shape.size(); i++) {
    if (std::find(dim.begin(), dim.end(), i) == dim.end() &&
        std::find(dim.begin(), dim.end(), i - x_shape.size()) == dim.end()) {
      dim_shape.push_back(x_shape[i]);
    }
  }

  if (keep_dim == true) {
    // reshape
    if (x_size == og_size) {
      paddle::platform::SetOutputNode(op, "X@GRAD", og, ngb_node_map);
      return;
    }
    auto og_dim = std::make_shared<ngraph::op::Reshape>(
        og, ngraph::AxisVector(axis_vector), ngraph::Shape(dim_shape));
    auto result =
        std::make_shared<ngraph::op::Broadcast>(og_dim, x_shape, axes);
    paddle::platform::SetOutputNode(op, "X@GRAD", result, ngb_node_map);

  } else {
    if (x_size == og_size) {
      auto og_dim = std::make_shared<ngraph::op::Reshape>(
          og, ngraph::AxisVector(axis_vector), x_shape);
      paddle::platform::SetOutputNode(op, "X@GRAD", og_dim, ngb_node_map);
    } else {
      if (og->get_shape().size() == 1 && og->get_shape()[0] == 1) {
        og = std::make_shared<ngraph::op::Reshape>(og, ngraph::AxisVector{0},
                                                   ngraph::Shape{});
      }
      auto result = std::make_shared<ngraph::op::Broadcast>(og, x_shape, axes);
      paddle::platform::SetOutputNode(op, "X@GRAD", result, ngb_node_map);
    }
  }
}
}  // namespace ngraphs
}  // namespace operators
}  // namespace paddle

REGISTER_NG_OP(reduce_sum, BuildReduceSumNode);
REGISTER_NG_OP(reduce_sum_grad, BuildReduceSumGradNode);
