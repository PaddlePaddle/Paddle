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

void BuildFillConstantNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto op_attrs = paddle::framework::AttrReader(op->Attrs());
  auto vsp = op_attrs.Get<std::vector<int64_t>>("shape");
  ngraph::Shape shape;
  for (auto& sp : vsp) {
    shape.push_back(sp);
  }
  float value = op_attrs.Get<float>("value");
  auto ng_dtype =
      platform::GetNgType(static_cast<paddle::framework::proto::VarType::Type>(
          op_attrs.Get<int>("dtype")));
  auto out = ngraph::op::Constant::create(ng_dtype, shape, {value});
  paddle::platform::SetOutputNode(op, "Out", out, ngb_node_map);
}
}  // namespace ngraphs
}  // namespace operators
}  // namespace paddle

REGISTER_NG_OP(fill_constant, BuildFillConstantNode);
