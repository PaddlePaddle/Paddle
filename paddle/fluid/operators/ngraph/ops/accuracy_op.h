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

void BuildAccuracyNode(
    const std::shared_ptr<framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto indices = platform::GetInputNode(op, "Indices", ngb_node_map);
  auto label = platform::GetInputNode(op, "Label", ngb_node_map);
  auto inference = platform::GetInputNode(op, "Out", ngb_node_map);
  auto inference_shape = inference->get_shape();
  size_t num_samples = inference_shape.at(0);
  size_t k = inference_shape.at(1);

  std::shared_ptr<ngraph::Node> label_k = label;
  if (k > 1) {
    auto label_1d = std::make_shared<ngraph::op::Reshape>(
        label, ngraph::AxisVector{0, 1}, ngraph::Shape{num_samples});
    label_k = std::make_shared<ngraph::op::Broadcast>(label_1d, inference_shape,
                                                      ngraph::AxisSet{1});
  }

  auto node_equal = std::make_shared<ngraph::op::Equal>(indices, label_k);
  auto node_eq_int =
      std::make_shared<ngraph::op::Convert>(node_equal, ngraph::element::i64);
  auto num_correct_0d =
      std::make_shared<ngraph::op::Sum>(node_eq_int, ngraph::AxisSet{0, 1});
  std::shared_ptr<ngraph::Node> num_correct =
      platform::NgReshaper(num_correct_0d, ngraph::Shape{1});
  std::shared_ptr<ngraph::Node> n_samples = ngraph::op::Constant::create(
      ngraph::element::i64, ngraph::Shape{1}, {num_samples});
  std::shared_ptr<ngraph::Node> accuracy = std::make_shared<ngraph::op::Divide>(
      std::make_shared<ngraph::op::Convert>(num_correct, ngraph::element::f32),
      std::make_shared<ngraph::op::Convert>(n_samples, ngraph::element::f32));

  platform::SetOutputNode(op, "Accuracy", accuracy, ngb_node_map);
  platform::SetOutputNode(op, "Correct", num_correct, ngb_node_map);
  platform::SetOutputNode(op, "Total", n_samples, ngb_node_map);
}
}  // namespace ngraphs
}  // namespace operators
}  // namespace paddle

REGISTER_NG_OP(accuracy, BuildAccuracyNode);
