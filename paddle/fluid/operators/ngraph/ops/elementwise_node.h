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

#include "ngraph/ngraph.hpp"
#include "paddle/fluid/operators/ngraph/ops/elementwise_binary_prepare_node.h"
#include "paddle/fluid/platform/ngraph_helper.h"

namespace paddle {
namespace operators {
namespace ngraphs {

template <typename T>
void BuildElementwiseBinaryNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto nodes = ElementwiseBinaryNodePrepare(op, ngb_node_map);
  std::shared_ptr<ngraph::Node>& x = nodes.at(0);
  std::shared_ptr<ngraph::Node>& y = nodes.at(1);

  if (x->get_element_type() != y->get_element_type()) {
    y = std::make_shared<ngraph::op::Convert>(y, x->get_element_type());
  }
  auto out = std::make_shared<T>(x, y);
  paddle::platform::SetOutputNode(op, "Out", out, ngb_node_map);
}

template <typename T>
void BuildElementwiseCompareNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto nodes = ElementwiseBinaryNodePrepare(op, ngb_node_map);
  std::shared_ptr<ngraph::Node>& x = nodes.at(0);
  std::shared_ptr<ngraph::Node>& y = nodes.at(1);

  if (x->get_element_type() != y->get_element_type()) {
    x = std::make_shared<ngraph::op::Convert>(x, ngraph::element::f64);
    y = std::make_shared<ngraph::op::Convert>(y, ngraph::element::f64);
  }
  auto out = std::make_shared<T>(x, y);
  paddle::platform::SetOutputNode(op, "Out", out, ngb_node_map);
}
}  // namespace ngraphs
}  // namespace operators
}  // namespace paddle
