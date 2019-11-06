// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "ngraph/ngraph.hpp"
#include "paddle/fluid/operators/ngraph/ops/op_bridge.h"
#include "paddle/fluid/platform/ngraph_helper.h"

namespace paddle {
namespace operators {
namespace ngraphs {
static void BuildLrnNode(
    const std::shared_ptr<framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto input = platform::GetInputNode(op, "X", ngb_node_map);

  auto op_attrs = framework::AttrReader(op->Attrs());
  const int n = op_attrs.Get<int>("n");
  const float alpha = op_attrs.Get<float>("alpha") * static_cast<float>(n);
  const float beta = op_attrs.Get<float>("beta");
  const float k = op_attrs.Get<float>("k");

  auto lrn_out = std::make_shared<ngraph::op::LRN>(input, alpha, beta, k, n);
  std::shared_ptr<ngraph::Node> mid_out = paddle::platform::CreateConstant(
      input->get_element_type(), input->get_shape(), {k});

  platform::SetOutputNode(op, "MidOut", mid_out, ngb_node_map);
  platform::SetOutputNode(op, "Out", lrn_out, ngb_node_map);
}

}  // namespace ngraphs
}  // namespace operators
}  // namespace paddle

REGISTER_NG_OP(lrn, BuildLrnNode);
