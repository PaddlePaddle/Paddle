// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/platform/device/ipu/popart_canonicalization/canonicalization_utils.h"
#include "paddle/fluid/platform/device/ipu/popart_canonicalization/op_builder.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace platform {
namespace ipu {
namespace {

Node *reduce_op_handler(Graph *graph, Node *node, const std::string &op_name) {
  auto *op = node->Op();
  auto attrs = AttributeMap{};
  auto reduce_all = BOOST_GET_CONST(bool, op->GetAttr("reduce_all"));
  if (!reduce_all) {
    auto axes_ = BOOST_GET_CONST(std::vector<int>, op->GetAttr("dim"));
    auto axes = std::vector<int64_t>{axes_.begin(), axes_.end()};
    attrs.emplace("axes", axes);
  }
  auto keepdims_ = BOOST_GET_CONST(bool, op->GetAttr("keep_dim"));
  auto keepdims = int64_t{keepdims_};
  attrs.emplace("keepdims", keepdims);
  return CreateBaseOp(graph, node, op_name, node->inputs, node->outputs, attrs);
}

Node *reduce_mean_handler(Graph *graph, Node *node) {
  return reduce_op_handler(graph, node, "popart_reducemean");
}

Node *reduce_min_handler(Graph *graph, Node *node) {
  return reduce_op_handler(graph, node, "popart_reducemin");
}

Node *reduce_sum_handler(Graph *graph, Node *node) {
  return reduce_op_handler(graph, node, "popart_reducesum");
}

Node *reduce_max_handler(Graph *graph, Node *node) {
  return reduce_op_handler(graph, node, "popart_reducemax");
}

Node *reduce_prod_handler(Graph *graph, Node *node) {
  return reduce_op_handler(graph, node, "popart_reduceprod");
}

}  // namespace
}  // namespace ipu
}  // namespace platform
}  // namespace paddle

REGISTER_HANDLER(reduce_mean, reduce_mean_handler);
REGISTER_HANDLER(reduce_min, reduce_min_handler);
REGISTER_HANDLER(reduce_sum, reduce_sum_handler);
REGISTER_HANDLER(reduce_max, reduce_max_handler);
REGISTER_HANDLER(reduce_prod, reduce_prod_handler);
