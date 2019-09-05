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

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "ngraph/ngraph.hpp"
#include "ngraph/op/experimental/generate_mask.hpp"
#include "paddle/fluid/operators/ngraph/ops/elementwise_scalar_op.h"
#include "paddle/fluid/operators/ngraph/ops/op_bridge.h"
#include "paddle/fluid/platform/ngraph_helper.h"

namespace paddle {
namespace operators {
namespace ngraphs {

static void BuildDropoutNode(
    const std::shared_ptr<framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto input = platform::GetInputNode(op, "X", ngb_node_map);
  auto op_attrs = framework::AttrReader(op->Attrs());
  auto dropout_prob = op_attrs.Get<float>("dropout_prob");
  auto dropout_implementation =
      op_attrs.Get<std::string>("dropout_implementation");
  auto is_test = op_attrs.Get<bool>("is_test");
  auto seed = op_attrs.Get<int>("seed");
  auto fix_seed = op_attrs.Get<bool>("fix_seed");
  float value = 1.0f - dropout_prob;
  bool upscale_in_train = (dropout_implementation == "upscale_in_train");

  if (is_test) {
    if (upscale_in_train) {
      platform::SetOutputNode(op, "Out", input, ngb_node_map);
    } else {
      auto mask_val = paddle::platform::CreateConstant(
          input->get_element_type(), input->get_shape(), {value});
      auto out = input * mask_val;
      platform::SetOutputNode(op, "Out", out, ngb_node_map);
    }
  } else {
    auto one = paddle::platform::CreateConstant(input->get_element_type(),
                                                ngraph::Shape{}, {1});

    auto gen_mask = std::make_shared<ngraph::op::GenerateMask>(
        one, input->get_shape(), input->get_element_type(), seed, value,
        fix_seed);

    if (upscale_in_train) {
      auto mask_val = paddle::platform::CreateConstant(
          input->get_element_type(), input->get_shape(), {value});

      auto out = value ? input * gen_mask / mask_val : input * gen_mask;
      platform::SetOutputNode(op, "Mask", gen_mask, ngb_node_map);
      platform::SetOutputNode(op, "Out", out, ngb_node_map);
    } else {
      auto out = input * gen_mask;
      platform::SetOutputNode(op, "Mask", gen_mask, ngb_node_map);
      platform::SetOutputNode(op, "Out", out, ngb_node_map);
    }
  }
}

static void BuildDropoutGradNode(
    const std::shared_ptr<framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto dy = platform::GetInputNode(op, "Out@GRAD", ngb_node_map);
  auto mask = platform::GetInputNode(op, "Mask", ngb_node_map);
  if (dy->get_element_type() != mask->get_element_type()) {
    mask = std::make_shared<ngraph::op::Convert>(mask, dy->get_element_type());
  }

  auto op_attrs = framework::AttrReader(op->Attrs());
  auto dropout_prob = op_attrs.Get<float>("dropout_prob");
  auto dropout_implementation =
      op_attrs.Get<std::string>("dropout_implementation");
  auto dx = dy * mask;

  if (dropout_implementation == "upscale_in_train") {
    if (dropout_prob == 1.0f) {
      dx = ElementwiseScalar<ngraph::op::Multiply>(0., dy);
    } else {
      dx =
          ElementwiseScalar<ngraph::op::Multiply>(1. / (1. - dropout_prob), dx);
    }
  }
  platform::SetOutputNode(op, "X@GRAD", dx, ngb_node_map);
}
}  // namespace ngraphs
}  // namespace operators
}  // namespace paddle

REGISTER_NG_OP(dropout, BuildDropoutNode);
REGISTER_NG_OP(dropout_grad, BuildDropoutGradNode);
