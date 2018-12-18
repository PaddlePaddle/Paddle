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

#include <algorithm>
#include <functional>
#include <vector>

#include "paddle/fluid/framework/ngraph_bridge.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/enforce.h"

#include "ngraph/ngraph.hpp"

namespace paddle {
namespace framework {

static std::shared_ptr<ngraph::Node> GetNode(
    const std::shared_ptr<OperatorBase>& op, const std::string name,
    const VariableNameMap& var_map,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto& var_names = var_map.at(name);
  PADDLE_ENFORCE_EQ(var_names.size(), 1,
                    "op %s name %s expects one associated var", op->Type(),
                    name);
  if (ngb_node_map->find(var_names[0]) != ngb_node_map->end()) {
    return (*ngb_node_map)[var_names[0]];
  } else {
    return nullptr;
  }
}

static std::shared_ptr<ngraph::Node> GetInputNode(
    const std::shared_ptr<OperatorBase>& op, const std::string name,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  return GetNode(op, name, op->Inputs(), ngb_node_map);
}

static std::shared_ptr<ngraph::Node> GetOutputNode(
    const std::shared_ptr<OperatorBase>& op, const std::string name,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  return GetNode(op, name, op->Outputs(), ngb_node_map);
}

static void SetOutputNode(
    const std::shared_ptr<OperatorBase>& op, const std::string name,
    std::shared_ptr<ngraph::Node> node,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto& var_names = op->Outputs().at(name);
  if (var_names.size() == 1) {
    (*ngb_node_map)[var_names[0]] = node;
  } else if (var_names.size() == 0) {
    (*ngb_node_map)[""] = node;
  } else {
    PADDLE_THROW("name %s has more than 1 var_names.", name);
  }
}

static bool HasOutput(const std::shared_ptr<OperatorBase>& op,
                      const std::string name) {
  auto& outputs = op->Outputs();
  if (outputs.find(name) == outputs.end()) return false;
  return outputs.at(name).size() > 0;
}

template <typename T>
static void BuildBinaryNode(
    const std::shared_ptr<OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto x = GetInputNode(op, "X", ngb_node_map);
  auto y = GetInputNode(op, "Y", ngb_node_map);
  auto out = std::make_shared<T>(x, y);
  SetOutputNode(op, "Out", out, ngb_node_map);
}

template <typename T>
static void BuildUnaryNode(
    const std::shared_ptr<OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto input = GetInputNode(op, "X", ngb_node_map);
  auto out = std::make_shared<T>(input);
  SetOutputNode(op, "Out", out, ngb_node_map);
}

std::map<std::string,
         std::function<void(const std::shared_ptr<OperatorBase>&,
                            std::shared_ptr<std::unordered_map<
                                std::string, std::shared_ptr<ngraph::Node>>>)>>
    NgraphBridge::NG_NODE_MAP = {{"relu", BuildUnaryNode<ngraph::op::Relu>},
                                 {"tanh", BuildUnaryNode<ngraph::op::Tanh>}};

void NgraphBridge::BuildNgNode(const std::shared_ptr<OperatorBase>& op) {
  auto& op_type = op->Type();
  NG_NODE_MAP[op_type](op, ngb_node_map_);
}

}  // namespace framework
}  // namespace paddle
