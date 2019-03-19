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
#include "paddle/fluid/operators/ngraph/ops/elementwise_scalar_op.h"
#include "paddle/fluid/operators/ngraph/ops/op_bridge.h"
#include "paddle/fluid/operators/ngraph/ops/op_helper.h"
#include "paddle/fluid/platform/ngraph_helper.h"

namespace paddle {
namespace operators {
namespace ngraphs {

void BuildSoftmaxNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto x = paddle::platform::GetInputNode(op, "X", ngb_node_map);
  auto softmax = GetSoftmax(x);
  paddle::platform::SetOutputNode(op, "Out", softmax, ngb_node_map);
}

void BuildSoftmaxGradNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto out = paddle::platform::GetInputNode(op, "Out", ngb_node_map);
  auto dout = paddle::platform::GetInputNode(op, "Out@GRAD", ngb_node_map);
  auto dx = GetSoftmaxGrad(out, dout);
  paddle::platform::SetOutputNode(op, "X@GRAD", dx, ngb_node_map);
}
}  // namespace ngraphs
}  // namespace operators
}  // namespace paddle

REGISTER_NG_OP(softmax, BuildSoftmaxNode);
REGISTER_NG_OP(softmax_grad, BuildSoftmaxGradNode);
