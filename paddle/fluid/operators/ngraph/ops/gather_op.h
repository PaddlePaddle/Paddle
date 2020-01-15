/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
  PADDLE_ENFORCE_NOT_NULL(x);

  auto index = platform::GetInputNode(op, "Index", ngb_node_map);
  auto& index_shape = index->get_shape();
  PADDLE_ENFORCE(index_shape.size() == 1 ||
                 (index_shape.size() == 2 && index_shape[1] == 1));
  if (index_shape.size() == 2) {
    index = platform::NgReshaper(index, ngraph::Shape{index_shape[0]});
  }

  auto out = std::make_shared<ngraph::op::Gather>(x, index);

  paddle::platform::SetOutputNode(op, "Out", out, ngb_node_map);
}
void BuildGatherGradNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto dout = platform::GetInputNode(op, "Out@GRAD", ngb_node_map);
  PADDLE_ENFORCE_NOT_NULL(dout);
  auto x = platform::GetInputNode(op, "X", ngb_node_map);

  auto index = platform::GetInputNode(op, "Index", ngb_node_map);
  auto& index_shape = index->get_shape();
  PADDLE_ENFORCE(index_shape.size() == 1 ||
                 (index_shape.size() == 2 && index_shape[1] == 1));
  if (index_shape.size() == 2) {
    index = platform::NgReshaper(index, ngraph::Shape{index_shape[0]});
  }

  std::shared_ptr<ngraph::Node> x0 = paddle::platform::CreateConstant(
      dout->get_element_type(), x->get_shape(), {0});
  auto dx = std::make_shared<ngraph::op::ScatterAdd>(x0, index, dout);
  paddle::platform::SetOutputNode(op, "X@GRAD", dx, ngb_node_map);
}
}  // namespace ngraphs
}  // namespace operators
}  // namespace paddle

REGISTER_NG_OP(gather, BuildGatherNode);
REGISTER_NG_OP(gather_grad, BuildGatherGradNode);
