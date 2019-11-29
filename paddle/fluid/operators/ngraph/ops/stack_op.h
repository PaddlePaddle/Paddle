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
#include "paddle/fluid/operators/ngraph/ops/op_bridge.h"
#include "paddle/fluid/platform/ngraph_helper.h"

namespace paddle {
namespace operators {
namespace ngraphs {

void BuildStackNode(
    const std::shared_ptr<framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto op_attrs = framework::AttrReader(op->Attrs());
  auto axis = op_attrs.Get<int>("axis");
  std::vector<std::shared_ptr<ngraph::Node>> args;
  for (auto& var_name_item : op->Inputs()) {
    for (auto& var_name : var_name_item.second) {
      auto& node = ngb_node_map->at(var_name);
      auto shape = node->get_shape();
      axis = (axis < 0) ? axis + shape.size() + 1 : axis;
      shape.insert(shape.begin() + axis, 1);
      std::vector<size_t> input_order(shape.size() - 1);
      std::iota(std::begin(input_order), std::end(input_order), 0);
      args.push_back(std::make_shared<ngraph::op::Reshape>(
          node, ngraph::AxisVector(input_order), shape));
    }
  }
  auto out = std::make_shared<ngraph::op::Concat>(args, axis);
  platform::SetOutputNode(op, "Y", out, ngb_node_map);
}
}  // namespace ngraphs
}  // namespace operators
}  // namespace paddle

REGISTER_NG_OP(stack, BuildStackNode);
