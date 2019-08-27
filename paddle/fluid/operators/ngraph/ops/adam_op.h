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

#include "ngraph/ngraph.hpp"
#include "paddle/fluid/operators/ngraph/ops/elementwise_scalar_op.h"
#include "paddle/fluid/operators/ngraph/ops/op_bridge.h"
#include "paddle/fluid/platform/ngraph_helper.h"

namespace paddle {
namespace operators {
namespace ngraphs {

void BuildAdamNode(
    const std::shared_ptr<framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto op_attrs = framework::AttrReader(op->Attrs());
  auto beta1pow = platform::GetInputNode(op, "Beta1Pow", ngb_node_map);
  auto beta2pow = platform::GetInputNode(op, "Beta2Pow", ngb_node_map);
  auto grad = platform::GetInputNode(op, "Grad", ngb_node_map);
  auto learning_rate = platform::GetInputNode(op, "LearningRate", ngb_node_map);
  auto moment1 = platform::GetInputNode(op, "Moment1", ngb_node_map);
  auto moment2 = platform::GetInputNode(op, "Moment2", ngb_node_map);
  auto param = platform::GetInputNode(op, "Param", ngb_node_map);

  auto epsilon = op_attrs.Get<float>("epsilon");
  auto beta2 = op_attrs.Get<float>("beta2");
  auto beta1 = op_attrs.Get<float>("beta1");

  auto moment1_shape = moment1->get_shape();
  auto grad_shape = grad->get_shape();

  auto moment1out = std::make_shared<ngraph::op::Add>(
      ElementwiseScalar<ngraph::op::Multiply>(beta1, moment1),
      ElementwiseScalar<ngraph::op::Multiply>(1. - beta1, grad));

  auto grad_square = std::make_shared<ngraph::op::Multiply>(grad, grad);
  auto moment2out = std::make_shared<ngraph::op::Add>(
      ElementwiseScalar<ngraph::op::Multiply>(beta2, moment2),
      ElementwiseScalar<ngraph::op::Multiply>(1. - beta2, grad_square));
  auto node_sqrt = std::make_shared<ngraph::op::Sqrt>(
      ElementwiseScalar<ngraph::op::Subtract>(1., beta2pow));
  auto lr = std::make_shared<ngraph::op::Divide>(
      node_sqrt, ElementwiseScalar<ngraph::op::Subtract>(1., beta1pow));
  auto updated_lr = std::make_shared<ngraph::op::Multiply>(learning_rate, lr);

  auto moment2_sqrt = std::make_shared<ngraph::op::Sqrt>(moment2out);
  auto param_grad = std::make_shared<ngraph::op::Divide>(
      moment1out, ElementwiseScalar<ngraph::op::Add>(epsilon, moment2_sqrt));
  auto delta = ElementwiseScalar<ngraph::op::Multiply>(updated_lr, param_grad);
  auto param_out = std::make_shared<ngraph::op::Subtract>(param, delta);

  platform::SetOutputNode(op, "Moment1Out", moment1out, ngb_node_map);
  platform::SetOutputNode(op, "Moment2Out", moment2out, ngb_node_map);
  platform::SetOutputNode(op, "ParamOut", param_out, ngb_node_map);
}
}  // namespace ngraphs
}  // namespace operators
}  // namespace paddle

REGISTER_NG_OP(adam, BuildAdamNode);
