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

void BuildConcatNode(
    const std::shared_ptr<framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  std::vector<std::shared_ptr<ngraph::Node>> args;
  for (auto& var_name_item : op->Inputs()) {
    for (auto& var_name : var_name_item.second) {
      auto& node0 = ngb_node_map->at(var_name);
      args.push_back(node0);
    }
  }
  auto op_attrs = framework::AttrReader(op->Attrs());
  const size_t axis = op_attrs.Get<int>("axis");
  auto out = std::make_shared<ngraph::op::Concat>(args, axis);
  platform::SetOutputNode(op, "Out", out, ngb_node_map);
}
}  // namespace ngraphs
}  // namespace operators
}  // namespace paddle

REGISTER_NG_OP(concat, BuildConcatNode);
