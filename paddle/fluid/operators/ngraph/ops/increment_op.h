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

#include <string>
#include <vector>

#include "ngraph/ngraph.hpp"
#include "paddle/fluid/operators/ngraph/ops/elementwise_node.h"
#include "paddle/fluid/platform/ngraph_helper.h"

namespace paddle {
namespace operators {
namespace ngraphs {

void BuildIncrementNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto x = paddle::platform::GetInputNode(op, "X", ngb_node_map);
  auto op_attrs = paddle::framework::AttrReader(op->Attrs());
  float step = op_attrs.Get<float>("step");
  auto step_op =
    std::make_shared<
        ngraph::op::Constant>(x->get_element_type(), x->get_shape(), std::vector<float>{step});
  std::shared_ptr<ngraph::Node> out =
      std::make_shared<ngraph::op::Add>(x, step_op);
  paddle::platform::SetOutputNode(op, "Out", out, ngb_node_map);
}

}  // namespace ngraphs
}  // namespace operators
}  // namespace paddle

REGISTER_NG_OP(increment, BuildIncrementNode);
