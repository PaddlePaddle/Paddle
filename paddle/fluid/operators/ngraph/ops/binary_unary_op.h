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

#include <memory>
#include <string>
#include <unordered_map>
#include "ngraph/ngraph.hpp"
#include "paddle/fluid/operators/ngraph/ops/op_bridge.h"
#include "paddle/fluid/platform/ngraph_helper.h"

namespace paddle {
namespace operators {
namespace ngraphs {

template <typename T>
static void BuildBinaryNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto x = paddle::platform::GetInputNode(op, "X", ngb_node_map);
  auto y = paddle::platform::GetInputNode(op, "Y", ngb_node_map);
  auto out = std::make_shared<T>(x, y);
  paddle::platform::SetOutputNode(op, "Out", out, ngb_node_map);
}

template <typename T>
static void BuildUnaryNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto input = paddle::platform::GetInputNode(op, "X", ngb_node_map);
  auto out = std::make_shared<T>(input);
  paddle::platform::SetOutputNode(op, "Out", out, ngb_node_map);
}
}  // namespace ngraphs
}  // namespace operators
}  // namespace paddle

REGISTER_NG_OP(abs, BuildUnaryNode<ngraph::op::Abs>);
REGISTER_NG_OP(relu, BuildUnaryNode<ngraph::op::Relu>);
REGISTER_NG_OP(tanh, BuildUnaryNode<ngraph::op::Tanh>);
REGISTER_NG_OP(sigmoid, BuildUnaryNode<ngraph::op::Sigmoid>);

REGISTER_NG_OP(logical_and, BuildBinaryNode<ngraph::op::And>);
REGISTER_NG_OP(logical_or, BuildBinaryNode<ngraph::op::Or>);
REGISTER_NG_OP(logical_not, BuildUnaryNode<ngraph::op::Not>);
