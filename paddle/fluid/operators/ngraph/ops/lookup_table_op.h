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
#include "ngraph/op/embedding_lookup.hpp"
#include "paddle/fluid/operators/lookup_table_op.h"
#include "paddle/fluid/operators/ngraph/ops/op_bridge.h"
#include "paddle/fluid/platform/ngraph_helper.h"

namespace paddle {
namespace operators {
namespace ngraphs {

void BuildLookupTableNode(
    const std::shared_ptr<framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto op_attrs = paddle::framework::AttrReader(op->Attrs());
  const bool is_sparse = op_attrs.Get<bool>("is_sparse");
  const int64_t padding_idx = op_attrs.Get<int64_t>("padding_idx");

  auto ng_ids = paddle::platform::GetInputNode(op, "Ids", ngb_node_map);
  PADDLE_ENFORCE_NOT_NULL(ng_ids);

  const auto ng_w = paddle::platform::GetInputNode(op, "W", ngb_node_map);
  PADDLE_ENFORCE_NOT_NULL(ng_w);

  if (is_sparse) {
    PADDLE_THROW("Sparsity is not yet supported in nGraph lookup_table op.");
  }

  if (padding_idx != kNoPadding) {
    PADDLE_THROW("Padding is not yet supported in nGraph lookup_table op.");
  }
  auto shape = ng_ids->get_shape();
  if (shape.back() == 1) {
    shape.pop_back();
    ng_ids = platform::NgReshaper(ng_ids, shape);
  }
  auto ng_lookup = std::make_shared<ngraph::op::Gather>(ng_w, ng_ids);
  platform::SetOutputNode(op, "Out", ng_lookup, ngb_node_map);
}

void BuildLookupTableGradNode(
    const std::shared_ptr<framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto op_attrs = paddle::framework::AttrReader(op->Attrs());
  const bool is_sparse = op_attrs.Get<bool>("is_sparse");
  const int64_t padding_idx = op_attrs.Get<int64_t>("padding_idx");

  auto ng_ids = paddle::platform::GetInputNode(op, "Ids", ngb_node_map);
  PADDLE_ENFORCE_NOT_NULL(ng_ids);

  const auto ng_w = paddle::platform::GetInputNode(op, "W", ngb_node_map);
  PADDLE_ENFORCE_NOT_NULL(ng_w);

  auto dout = paddle::platform::GetInputNode(op, "Out@GRAD", ngb_node_map);

  if (is_sparse) {
    PADDLE_THROW("Sparsity is not yet supported in nGraph lookup_table op.");
  }

  if (padding_idx != kNoPadding) {
    PADDLE_THROW("Padding is not yet supported in nGraph lookup_table op.");
  }
  auto shape = ng_ids->get_shape();
  if (shape.back() == 1) {
    shape.pop_back();
    ng_ids = platform::NgReshaper(ng_ids, shape);
  }

  std::shared_ptr<ngraph::Node> W0 = paddle::platform::CreateConstant(
      dout->get_element_type(), ng_w->get_shape(), {0});
  auto dW = std::make_shared<ngraph::op::ScatterAdd>(W0, ng_ids, dout);
  platform::SetOutputNode(op, "W@GRAD", dW, ngb_node_map);
}
}  // namespace ngraphs
}  // namespace operators
}  // namespace paddle

REGISTER_NG_OP(lookup_table, BuildLookupTableNode);
REGISTER_NG_OP(lookup_table_grad, BuildLookupTableGradNode);
