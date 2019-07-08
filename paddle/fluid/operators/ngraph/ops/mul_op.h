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
#include "ngraph/ngraph.hpp"
#include "paddle/fluid/operators/ngraph/ops/op_bridge.h"
#include "paddle/fluid/platform/ngraph_helper.h"

namespace paddle {
namespace operators {
namespace ngraphs {

static void BuildMulNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto op_attrs = paddle::framework::AttrReader(op->Attrs());
  int x_num_col_dims = op_attrs.Get<int>("x_num_col_dims");
  int y_num_col_dims = op_attrs.Get<int>("y_num_col_dims");
  auto x = paddle::platform::GetInputNode(op, "X", ngb_node_map);
  auto y = paddle::platform::GetInputNode(op, "Y", ngb_node_map);
  int y_rank = y->get_shape().size();

  auto x_reshape = x;
  auto y_reshape = y;

  if (x->get_shape().size() > 2) {
    auto x_2d = paddle::platform::FlattenTo2d(x->get_shape(), x_num_col_dims);
    x_reshape = paddle::platform::NgReshaper(x, x_2d);
  }

  if (y->get_shape().size() > 2) {
    auto y_2d = paddle::platform::FlattenTo2d(y->get_shape(), y_num_col_dims);
    y_reshape = paddle::platform::NgReshaper(y, y_2d);
  }

  std::shared_ptr<ngraph::Node> out =
      std::make_shared<ngraph::op::Dot>(x_reshape, y_reshape);

  ngraph::Shape out_shape;
  for (int i = 0; i < x_num_col_dims; ++i) {
    out_shape.push_back(x->get_shape()[i]);
  }
  for (int i = y_num_col_dims; i < y_rank; ++i) {
    out_shape.push_back(y->get_shape()[i]);
  }
  out = paddle::platform::NgReshaper(out, out_shape);
  paddle::platform::SetOutputNode(op, "Out", out, ngb_node_map);
}

static void BuildMulGradNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto op_attrs = paddle::framework::AttrReader(op->Attrs());
  int x_num_col_dims = op_attrs.Get<int>("x_num_col_dims");
  int y_num_col_dims = op_attrs.Get<int>("y_num_col_dims");
  auto x = paddle::platform::GetInputNode(op, "X", ngb_node_map);
  auto y = paddle::platform::GetInputNode(op, "Y", ngb_node_map);
  auto dout = paddle::platform::GetInputNode(op, "Out@GRAD", ngb_node_map);

  bool is_dx = paddle::platform::HasOutput(op, "X@GRAD") ? true : false;
  bool is_dy = paddle::platform::HasOutput(op, "Y@GRAD") ? true : false;

  auto x_shape = x->get_shape();
  auto y_shape = y->get_shape();

  auto x_reshape = x;
  auto y_reshape = y;

  if (x_shape.size() > 2) {
    auto x_2d_shape = paddle::platform::FlattenTo2d(x_shape, x_num_col_dims);
    x_reshape = paddle::platform::NgReshaper(x, x_2d_shape);
  }

  if (y_shape.size() > 2) {
    auto y_2d_shape = paddle::platform::FlattenTo2d(y_shape, y_num_col_dims);
    y_reshape = paddle::platform::NgReshaper(y, y_2d_shape);
  }

  auto x_reshape_shape = x_reshape->get_shape();
  std::reverse(x_reshape_shape.begin(), x_reshape_shape.end());
  auto x_transpose = std::make_shared<ngraph::op::Reshape>(
      x_reshape, ngraph::AxisVector{1, 0}, x_reshape_shape);

  auto y_reshape_shape = y_reshape->get_shape();
  std::reverse(y_reshape_shape.begin(), y_reshape_shape.end());
  auto y_transpose = std::make_shared<ngraph::op::Reshape>(
      y_reshape, ngraph::AxisVector{1, 0}, y_reshape_shape);

  if (is_dx) {
    if (dout->get_shape().size() > 2) {
      auto dout_2d_shape = paddle::platform::FlattenTo2d(dout->get_shape(), 2);
      dout = paddle::platform::NgReshaper(dout, dout_2d_shape);
    }
    auto dx = std::make_shared<ngraph::op::Dot>(dout, y_transpose);

    if (dx->get_shape() == x_shape) {
      paddle::platform::SetOutputNode(op, "X@GRAD", dx, ngb_node_map);
    } else {
      auto dx_reshape = paddle::platform::NgReshaper(dx, x_shape);
      paddle::platform::SetOutputNode(op, "X@GRAD", dx_reshape, ngb_node_map);
    }
  }

  if (is_dy) {
    if (dout->get_shape().size() > 2) {
      auto dout_2d_shape = paddle::platform::FlattenTo2d(dout->get_shape(), 2);
      dout = paddle::platform::NgReshaper(dout, dout_2d_shape);
    }
    auto dy = std::make_shared<ngraph::op::Dot>(x_transpose, dout);

    if (dy->get_shape() == y_shape) {
      paddle::platform::SetOutputNode(op, "Y@GRAD", dy, ngb_node_map);
    } else {
      auto dy_reshape = paddle::platform::NgReshaper(dy, y_shape);
      paddle::platform::SetOutputNode(op, "Y@GRAD", dy_reshape, ngb_node_map);
    }
  }
}
}  // namespace ngraphs
}  // namespace operators
}  // namespace paddle

REGISTER_NG_OP(mul, BuildMulNode);
REGISTER_NG_OP(mul_grad, BuildMulGradNode);
