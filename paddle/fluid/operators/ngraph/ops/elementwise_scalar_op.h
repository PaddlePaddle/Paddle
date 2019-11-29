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
#include "paddle/fluid/platform/ngraph_helper.h"

namespace paddle {
namespace operators {
namespace ngraphs {

template <typename T>
std::shared_ptr<ngraph::Node> ElementwiseScalar(
    float scale, std::shared_ptr<ngraph::Node> node) {
  auto node_shape = node->get_shape();
  auto scale_const = ngraph::op::Constant::create(node->get_element_type(),
                                                  node_shape, {scale});
  return std::make_shared<T>(scale_const, node);
}

template <typename T>
std::shared_ptr<ngraph::Node> ElementwiseScalar(
    std::shared_ptr<ngraph::Node> scale_1d,
    std::shared_ptr<ngraph::Node> node) {
  auto scale_shape = scale_1d->get_shape();
  PADDLE_ENFORCE_EQ(scale_shape.size(), 1, "Supporting 1d scale node");
  PADDLE_ENFORCE_EQ(scale_shape.at(0), 1, "scale 1d in in shape {1}");

  auto node_shape = node->get_shape();
  ngraph::AxisSet axis_set;
  for (size_t i = 0; i < node_shape.size(); ++i) {
    axis_set.insert(i);
  }
  node_shape.push_back(1);

  auto scale_bcast =
      std::make_shared<ngraph::op::Broadcast>(scale_1d, node_shape, axis_set);

  auto scale_reshape =
      paddle::platform::NgReshaper(scale_bcast, node->get_shape());

  return std::make_shared<T>(scale_reshape, node);
}
}  // namespace ngraphs
}  // namespace operators
}  // namespace paddle
