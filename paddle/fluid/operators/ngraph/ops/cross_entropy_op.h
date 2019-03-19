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

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "ngraph/ngraph.hpp"
#include "paddle/fluid/operators/ngraph/ops/op_bridge.h"
#include "paddle/fluid/operators/ngraph/ops/op_helper.h"
#include "paddle/fluid/platform/ngraph_helper.h"

namespace paddle {
namespace operators {
namespace ngraphs {

void BuildCrossEntropyNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto x = paddle::platform::GetInputNode(op, "X", ngb_node_map);
  auto label = paddle::platform::GetInputNode(op, "Label", ngb_node_map);
  auto op_attrs = paddle::framework::AttrReader(op->Attrs());
  const bool is_soft_label = op_attrs.Get<bool>("soft_label");
  int ignore_index = op_attrs.Get<int>("ignore_index");
  auto xe = GetCrossEntropy(x, label, is_soft_label, ignore_index);
  paddle::platform::SetOutputNode(op, "Y", xe, ngb_node_map);
}

void BuildCrossEntropyGradNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto op_attrs = paddle::framework::AttrReader(op->Attrs());
  const bool is_soft_label = op_attrs.Get<bool>("soft_label");
  int ignore_index = op_attrs.Get<int>("ignore_index");
  auto x = paddle::platform::GetInputNode(op, "X", ngb_node_map);
  auto label = paddle::platform::GetInputNode(op, "Label", ngb_node_map);
  auto dy = paddle::platform::GetInputNode(op, "Y@GRAD", ngb_node_map);
  auto xe_grad = GetCrossEntropyGrad(x, label, dy, is_soft_label, ignore_index);
  paddle::platform::SetOutputNode(op, "X@GRAD", xe_grad, ngb_node_map);
}
}  // namespace ngraphs
}  // namespace operators
}  // namespace paddle

REGISTER_NG_OP(cross_entropy, BuildCrossEntropyNode);
REGISTER_NG_OP(cross_entropy_grad, BuildCrossEntropyGradNode);
