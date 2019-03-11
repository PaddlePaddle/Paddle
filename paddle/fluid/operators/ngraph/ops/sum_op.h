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

#include <string>
#include <vector>

#include "ngraph/ngraph.hpp"
#include "paddle/fluid/operators/ngraph/ops/op_bridge.h"
#include "paddle/fluid/platform/ngraph_helper.h"

namespace paddle {
namespace operators {
namespace ngraphs {

void BuildSumNode(
    const std::shared_ptr<framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  std::vector<std::string> op_inputs;
  for (auto& var_name_item : op->Inputs()) {
    for (auto& var_name : var_name_item.second) {
      op_inputs.push_back(var_name);
      if (ngb_node_map->find(var_name) == ngb_node_map->end()) {
        PADDLE_THROW("op % input varname %s is not found in var_node_map",
                     op->Type(), var_name);
      }
    }
  }
  std::shared_ptr<ngraph::Node>& sum = ngb_node_map->at(op_inputs[0]);
  for (size_t k = 1; k < op_inputs.size(); ++k) {
    std::shared_ptr<ngraph::Node>& nodek = ngb_node_map->at(op_inputs[k]);
    if (nodek->get_element_type() != sum->get_element_type()) {
      nodek =
          std::make_shared<ngraph::op::Convert>(nodek, sum->get_element_type());
    }
    sum = sum + nodek;
  }
  platform::SetOutputNode(op, "Out", sum, ngb_node_map);
}
}  // namespace ngraphs
}  // namespace operators
}  // namespace paddle

REGISTER_NG_OP(sum, BuildSumNode);
