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
#include "paddle/fluid/operators/ngraph/ops/cross_entropy_op.h"
#include "paddle/fluid/operators/ngraph/ops/softmax_op.h"
#include "paddle/fluid/platform/ngraph_helper.h"

namespace paddle {
namespace operators {
namespace ngraphs {

void BuildSoftmaxWithCrossEntropyNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto logits = paddle::platform::GetInputNode(op, "Logits", ngb_node_map);
  auto label = paddle::platform::GetInputNode(op, "Label", ngb_node_map);
  auto softmax = paddle::operators::ngraphs::GetSoftmax(logits);

  auto op_attrs = framework::AttrReader(op->Attrs());
  const bool is_soft_label = op_attrs.Get<bool>("soft_label");
  int ignore_index = op_attrs.Get<int>("ignore_index");
  auto xe = paddle::operators::ngraphs::GetCrossEntropy(
      softmax, label, is_soft_label, ignore_index);

  paddle::platform::SetOutputNode(op, "Softmax", softmax, ngb_node_map);
  paddle::platform::SetOutputNode(op, "Loss", xe, ngb_node_map);
}

void BuildSoftmaxWithCrossEntropyGradNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto op_attrs = framework::AttrReader(op->Attrs());
  const bool is_soft_label = op_attrs.Get<bool>("soft_label");
  auto label = paddle::platform::GetInputNode(op, "Label", ngb_node_map);
  auto softmax = paddle::platform::GetInputNode(op, "Softmax", ngb_node_map);
  auto loss_grad =
      paddle::platform::GetInputNode(op, "Loss@GRAD", ngb_node_map);
  auto softmax_shape = softmax->get_shape();
  auto rank = softmax_shape.size();
  if (!is_soft_label) {
    auto label_shape = label->get_shape();
    label_shape.pop_back();
    label = platform::NgReshaper(label, label_shape);

    label =
        std::make_shared<ngraph::op::OneHot>(label, softmax_shape, rank - 1);
  }

  auto loss_grad_shape = loss_grad->get_shape();
  loss_grad_shape.pop_back();
  auto loss_grad_reshape = platform::NgReshaper(loss_grad, loss_grad_shape);
  auto loss_grad_bcast = std::make_shared<ngraph::op::Broadcast>(
      loss_grad_reshape, softmax_shape, ngraph::AxisSet{rank - 1});
  if (softmax->get_element_type() != label->get_element_type()) {
    label = std::make_shared<ngraph::op::Convert>(label,
                                                  softmax->get_element_type());
  }

  auto logits_grad = loss_grad_bcast * (softmax - label);
  paddle::platform::SetOutputNode(op, "Logits@GRAD", logits_grad, ngb_node_map);
}
}  // namespace ngraphs
}  // namespace operators
}  // namespace paddle

REGISTER_NG_OP(softmax_with_cross_entropy, BuildSoftmaxWithCrossEntropyNode);
REGISTER_NG_OP(softmax_with_cross_entropy_grad,
               BuildSoftmaxWithCrossEntropyGradNode);
