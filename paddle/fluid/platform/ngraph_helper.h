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

#ifdef PADDLE_WITH_NGRAPH
#pragma once

#include <functional>
#include <string>
#include <vector>
#include "ngraph/ngraph.hpp"

namespace paddle {
namespace platform {

static ngraph::Shape FlattenTo2d(ngraph::Shape sh, int num) {
  auto x1 = std::accumulate(std::begin(sh), std::begin(sh) + num, 1,
                            std::multiplies<size_t>());
  auto x2 = std::accumulate(std::begin(sh) + num, std::end(sh), 1,
                            std::multiplies<size_t>());
  size_t x1_l = static_cast<size_t>(x1);
  size_t x2_l = static_cast<size_t>(x2);
  return ngraph::Shape{x1_l, x2_l};
}

static std::shared_ptr<ngraph::Node> NgReshaper(
    std::shared_ptr<ngraph::Node> input, ngraph::Shape shape) {
  std::vector<size_t> input_order(input->get_shape().size());
  std::iota(std::begin(input_order), std::end(input_order), 0);
  return std::make_shared<ngraph::op::Reshape>(
      input, ngraph::AxisVector(input_order), shape);
}

static std::shared_ptr<ngraph::Node> GetNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    const std::string prm, const paddle::framework::VariableNameMap& var_map,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto& var_names = var_map.at(prm);
  PADDLE_ENFORCE_EQ(var_names.size(), 1,
                    "op %s prm %s expects one associated var", op->Type(), prm);
  if (ngb_node_map->find(var_names[0]) != ngb_node_map->end()) {
    return (*ngb_node_map)[var_names[0]];
  } else {
    return nullptr;
  }
}

static std::shared_ptr<ngraph::Node> GetInputNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    const std::string prm,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  return GetNode(op, prm, op->Inputs(), ngb_node_map);
}

static std::shared_ptr<ngraph::Node> GetOutputNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    const std::string prm,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  return GetNode(op, prm, op->Outputs(), ngb_node_map);
}

static void SetOutputNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    const std::string prm, std::shared_ptr<ngraph::Node> node,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto& var_names = op->Outputs().at(prm);
  if (var_names.size() == 1) {
    (*ngb_node_map)[var_names[0]] = node;
  } else if (var_names.size() == 0) {
    (*ngb_node_map)[""] = node;
  } else {
    PADDLE_THROW("prm %s has more than 1 var_names.", prm);
  }
}

static bool HasOutput(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    const std::string prm) {
  auto& outputs = op->Outputs();
  if (outputs.find(prm) == outputs.end()) return false;
  return outputs.at(prm).size() > 0;
}

}  // namespace platform
}  // namespace paddle

#endif
