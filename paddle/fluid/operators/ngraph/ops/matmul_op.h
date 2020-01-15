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
#include "paddle/fluid/operators/ngraph/ops/op_bridge.h"
#include "paddle/fluid/platform/ngraph_helper.h"

namespace paddle {
namespace operators {
namespace ngraphs {

std::shared_ptr<ngraph::Node> transposeAndFlat3D(
    const std::shared_ptr<ngraph::Node>& input, const bool transpose,
    bool x = true) {
  auto shape = input->get_shape();
  size_t n = shape.size();
  std::shared_ptr<ngraph::Node> output;
  if (n >= 3) {
    std::vector<size_t> order(n);
    std::iota(std::begin(order), std::end(order), 0);
    size_t outer = 1;
    for (size_t i = 0; i < n - 2; i++) {
      outer = outer * shape[i];
    }
    std::vector<size_t> reshape{outer, shape[n - 2], shape[n - 1]};

    if (transpose == true) {
      order[n - 2] = n - 1;
      order[n - 1] = n - 2;
      reshape[2] = shape[n - 2];
      reshape[1] = shape[n - 1];
    }
    output = std::make_shared<ngraph::op::Reshape>(
        input, ngraph::AxisVector(order), ngraph::Shape(reshape));
  } else {
    std::shared_ptr<ngraph::Node> temp;
    if (n == 1 && x == true) {
      temp = std::make_shared<ngraph::op::Reshape>(input, ngraph::AxisVector{0},
                                                   ngraph::Shape{1, shape[0]});
    } else if (n == 1 && x == false) {
      temp = std::make_shared<ngraph::op::Reshape>(input, ngraph::AxisVector{0},
                                                   ngraph::Shape{shape[0], 1});
    } else {
      temp = input;
    }
    auto temp_shape = temp->get_shape();
    if (transpose == true) {
      output = std::make_shared<ngraph::op::Reshape>(
          temp, ngraph::AxisVector{1, 0},
          ngraph::Shape{temp_shape[1], temp_shape[0]});
    } else {
      output = temp;
    }
  }
  return output;
}
std::shared_ptr<ngraph::Node> broadcast3D(
    const std::shared_ptr<ngraph::Node>& input, size_t axis0) {
  auto shape = input->get_shape();
  size_t n = shape.size();
  if (n == 2) {
    auto output = std::make_shared<ngraph::op::Broadcast>(
        input, ngraph::Shape{axis0, shape[0], shape[1]}, ngraph::AxisSet{0});
    return output;
  }
  return input;
}
std::shared_ptr<ngraph::Node> dotOp(const std::shared_ptr<ngraph::Node>& a,
                                    const std::shared_ptr<ngraph::Node>& b) {
  std::shared_ptr<ngraph::Node> out;
  auto a_shape = a->get_shape();
  auto na = a_shape.size();
  auto b_shape = b->get_shape();
  auto nb = b_shape.size();
  if (na > 2 && nb > 2) {
    out = std::make_shared<ngraph::op::BatchMatMul>(a, b);
  } else {
    out = std::make_shared<ngraph::op::Dot>(a, b);
  }
  return out;
}
std::shared_ptr<ngraph::Node> reshapeToOriginal(
    std::shared_ptr<ngraph::Node> input, const ngraph::Shape& shape) {
  auto input_shape = input->get_shape();
  std::vector<size_t> axis(input_shape.size());
  std::iota(axis.begin(), axis.end(), 0);
  auto out = std::make_shared<ngraph::op::Reshape>(input, axis, shape);
  return out;
}
void BuildMatMulNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto x = paddle::platform::GetInputNode(op, "X", ngb_node_map);
  auto y = paddle::platform::GetInputNode(op, "Y", ngb_node_map);
  auto op_attrs = paddle::framework::AttrReader(op->Attrs());
  bool transpose_x = op_attrs.Get<bool>("transpose_X");
  bool transpose_y = op_attrs.Get<bool>("transpose_Y");
  float alpha = op_attrs.Get<float>("alpha");

  std::shared_ptr<ngraph::Node> out;
  auto x_shape = x->get_shape();
  auto y_shape = y->get_shape();
  size_t nx = x_shape.size();
  size_t ny = y_shape.size();
  x = transposeAndFlat3D(x, transpose_x, true);
  y = transposeAndFlat3D(y, transpose_y, false);
  auto y_shape3 = y->get_shape();
  auto x_shape3 = x->get_shape();
  if (nx > 2 || ny > 2) {
    ngraph::Shape out_shape = x_shape;
    if (nx != 3) {
      x = broadcast3D(x, y_shape3[0]);
      out_shape = y_shape;
    }
    if (ny != 3) {
      y = broadcast3D(y, x_shape3[0]);
      out_shape = x_shape;
    }
    auto nout = out_shape.size();
    auto out3 = std::make_shared<ngraph::op::BatchMatMul>(x, y);
    auto out3_shape = out3->get_shape();
    out_shape[nout - 1] = out3_shape[2];
    out_shape[nout - 2] = out3_shape[1];
    out = std::make_shared<ngraph::op::Reshape>(
        out3, ngraph::AxisVector{0, 1, 2}, out_shape);
  } else {
    out = std::make_shared<ngraph::op::Dot>(x, y);
  }
  auto out_shape = out->get_shape();
  std::vector<size_t> axis(out_shape.size());
  std::iota(axis.begin(), axis.end(), 0);
  for (size_t i = out_shape.size() - 1; i > 0; i--) {
    if (out_shape[i] == 1) {
      out_shape.erase(out_shape.begin() + i);
    }
  }
  auto out_ = std::make_shared<ngraph::op::Reshape>(
      out, ngraph::AxisVector(axis), out_shape);
  auto out_alpha = ElementwiseScalar<ngraph::op::Multiply>(alpha, out_);
  paddle::platform::SetOutputNode(op, "Out", out_alpha, ngb_node_map);
}

void BuildMatMulGradNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto op_attrs = paddle::framework::AttrReader(op->Attrs());

  auto dout = paddle::platform::GetInputNode(op, "Out@GRAD", ngb_node_map);
  auto y = paddle::platform::GetInputNode(op, "Y", ngb_node_map);
  auto x = paddle::platform::GetInputNode(op, "X", ngb_node_map);

  bool is_dx = paddle::platform::HasOutput(op, "X@GRAD") ? true : false;
  bool is_dy = paddle::platform::HasOutput(op, "Y@GRAD") ? true : false;
  bool transpose_x = op_attrs.Get<bool>("transpose_X");
  bool transpose_y = op_attrs.Get<bool>("transpose_Y");
  float alpha = op_attrs.Get<float>("alpha");
  auto dout_shape = dout->get_shape();
  auto x_shape = x->get_shape();
  auto y_shape = y->get_shape();
  size_t nx = x_shape.size();
  size_t ny = y_shape.size();
  size_t ndout = dout_shape.size();
  std::shared_ptr<ngraph::Node> x2, y2;
  std::shared_ptr<ngraph::Node> dout2;

  x2 = transposeAndFlat3D(x, false);
  y2 = transposeAndFlat3D(y, false, false);
  dout2 = transposeAndFlat3D(dout, false);
  auto x2_shape = x2->get_shape();
  auto y2_shape = y2->get_shape();
  if (nx >= 3 || ny >= 3) {
    std::shared_ptr<ngraph::Node> dout_temp;
    if (ndout == 2) {
      dout_temp = std::make_shared<ngraph::op::Reshape>(
          dout, ngraph::AxisVector{0, 1},
          ngraph::Shape{dout_shape[0], dout_shape[1], 1});
      if (ny < 3) {
        dout2 = dout_temp;
      } else {
        dout2 = transposeAndFlat3D(dout_temp, true);
      }
    }
    x2 = broadcast3D(x2, y_shape[0]);
    y2 = broadcast3D(y2, x_shape[0]);

  } else {
    dout2 = transposeAndFlat3D(dout, false, nx == 1 && transpose_x == false);
  }

  if (transpose_y == false) {
    y2 = transposeAndFlat3D(y2, true);
  }
  if (transpose_x == false) {
    x2 = transposeAndFlat3D(x2, true);
  }
  auto dx = dotOp(dout2, y2);
  auto dy = dotOp(x2, dout2);
  if (transpose_x == true) {
    dx = transposeAndFlat3D(dx, true);
  }
  if (transpose_y == true) {
    dy = transposeAndFlat3D(dy, true);
  }

  if (nx < 3 && ny >= 3) {
    dx = std::make_shared<ngraph::op::Sum>(dx, ngraph::AxisSet{0});
  }
  if (ny < 3 && nx >= 3) {
    dy = std::make_shared<ngraph::op::Sum>(dy, ngraph::AxisSet{0});
  }
  auto dx_t = reshapeToOriginal(dx, x_shape);
  auto dy_t = reshapeToOriginal(dy, y_shape);
  auto dx_scale = ElementwiseScalar<ngraph::op::Multiply>(1 / alpha, dx_t);
  auto dy_scale = ElementwiseScalar<ngraph::op::Multiply>(1 / alpha, dy_t);
  if (is_dx)
    paddle::platform::SetOutputNode(op, "X@GRAD", dx_scale, ngb_node_map);
  if (is_dy)
    paddle::platform::SetOutputNode(op, "Y@GRAD", dy_scale, ngb_node_map);
}
}  // namespace ngraphs
}  // namespace operators
}  // namespace paddle

REGISTER_NG_OP(matmul, BuildMatMulNode);
REGISTER_NG_OP(matmul_grad, BuildMatMulGradNode);
