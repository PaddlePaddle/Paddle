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

template <bool is_v2>
static void BuildTransposeNode(
    const std::shared_ptr<framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto input = platform::GetInputNode(op, "X", ngb_node_map);
  auto op_attrs = framework::AttrReader(op->Attrs());
  std::vector<int> axis = op_attrs.Get<std::vector<int>>("axis");

  auto input_shape = input->get_shape();
  ngraph::Shape x_reshape_shape;
  ngraph::AxisVector axis_vec;
  for (auto& v : axis) {
    axis_vec.push_back(v);
    x_reshape_shape.push_back(input_shape[v]);
  }
  std::shared_ptr<ngraph::Node> x_transpose =
      std::make_shared<ngraph::op::Reshape>(input, axis_vec, input_shape);
  x_transpose = platform::NgReshaper(x_transpose, x_reshape_shape);
  platform::SetOutputNode(op, "Out", x_transpose, ngb_node_map);
  if (is_v2) {
    ngraph::Shape input_xshape(input_shape.size() + 1);
    input_xshape[0] = 0;
    std::copy(input_shape.begin(), input_shape.end(), input_xshape.begin() + 1);
    auto xshape_node = std::make_shared<ngraph::op::Constant>(
        input->get_element_type(), input_xshape, std::vector<std::string>{});
    platform::SetOutputNode(op, "XShape", xshape_node, ngb_node_map);
  }
}

template <bool is_v2>
static void BuildTransposeGradNode(
    const std::shared_ptr<framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto input = platform::GetInputNode(op, "Out@GRAD", ngb_node_map);
  auto op_attrs = framework::AttrReader(op->Attrs());
  std::vector<int> axis = op_attrs.Get<std::vector<int>>("axis");

  ngraph::AxisVector axis_vec(axis.size());
  for (size_t i = 0; i < axis.size(); ++i) {
    axis_vec[axis.at(i)] = i;
  }

  ngraph::Shape out_shape;
  if (is_v2) {
    auto& xshape =
        platform::GetInputNode(op, "XShape", ngb_node_map)->get_shape();
    out_shape.resize(xshape.size() - 1);
    std::copy(xshape.begin() + 1, xshape.end(), out_shape.begin());
  } else {
    out_shape = platform::GetInputNode(op, "X", ngb_node_map)->get_shape();
  }

  std::shared_ptr<ngraph::Node> x_transpose =
      std::make_shared<ngraph::op::Reshape>(input, axis_vec, out_shape);

  platform::SetOutputNode(op, "X@GRAD", x_transpose, ngb_node_map);
}

}  // namespace ngraphs
}  // namespace operators
}  // namespace paddle

REGISTER_NG_OP(transpose, BuildTransposeNode<false>);
REGISTER_NG_OP(transpose_grad, BuildTransposeGradNode<false>);
REGISTER_NG_OP(transpose2, BuildTransposeNode<true>);
REGISTER_NG_OP(transpose2_grad, BuildTransposeGradNode<true>);
