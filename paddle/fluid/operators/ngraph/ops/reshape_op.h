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
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/ngraph_helper.h"

namespace paddle {
namespace operators {
namespace ngraphs {

ngraph::Shape calc_output_shape(const ngraph::Shape& input_shape,
                                const std::vector<int>& v_shape) {
  auto out_shape = v_shape;
  for (size_t i = 0; i < v_shape.size(); ++i) {
    if (v_shape[i] == 0) {
      out_shape[i] = input_shape[i];
    }
  }
  int size_input = ngraph::shape_size(input_shape);
  int size_out = 1;
  for (auto o : out_shape) {
    if (o > 0) size_out *= o;
  }
  for (auto& o : out_shape) {
    if (o == -1) o = size_input / size_out;
  }
  return ngraph::Shape(out_shape.begin(), out_shape.end());
}

template <bool is_v2>
static void BuildReshapeNode(
    const std::shared_ptr<framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  std::shared_ptr<ngraph::Node> input =
      platform::GetInputNode(op, "X", ngb_node_map);
  auto input_shape = input->get_shape();

  std::shared_ptr<ngraph::Node> shape =
      platform::GetInputNode(op, "Shape", ngb_node_map);
  PADDLE_ENFORCE_EQ(shape, nullptr,
                    platform::errors::Unimplemented(
                        "Support for Shape input is not implemented"));

  auto op_attrs = framework::AttrReader(op->Attrs());
  std::vector<int> v_shape = op_attrs.Get<std::vector<int>>("shape");

  auto out_shape = calc_output_shape(input_shape, v_shape);
  auto out = platform::NgReshaper(input, out_shape);
  platform::SetOutputNode(op, "Out", out, ngb_node_map);

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
void BuildReshapeGradNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto dout = paddle::platform::GetInputNode(op, "Out@GRAD", ngb_node_map);
  ngraph::Shape out_shape;
  if (is_v2) {
    auto& xshape =
        platform::GetInputNode(op, "XShape", ngb_node_map)->get_shape();
    out_shape.resize(xshape.size() - 1);
    std::copy(xshape.begin() + 1, xshape.end(), out_shape.begin());
  } else {
    auto input = paddle::platform::GetInputNode(op, "X", ngb_node_map);
    out_shape = input->get_shape();
  }
  auto dx = platform::NgReshaper(dout, out_shape);
  paddle::platform::SetOutputNode(op, "X@GRAD", dx, ngb_node_map);
}
}  // namespace ngraphs
}  // namespace operators
}  // namespace paddle

REGISTER_NG_OP(reshape, BuildReshapeNode<false>);
REGISTER_NG_OP(reshape2, BuildReshapeNode<true>);
REGISTER_NG_OP(reshape_grad, BuildReshapeGradNode<false>);
REGISTER_NG_OP(reshape2_grad, BuildReshapeGradNode<true>);
