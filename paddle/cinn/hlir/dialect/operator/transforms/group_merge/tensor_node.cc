// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#pragma once

#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/tensor_node.h"

#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/op_node.h"

namespace cinn {
namespace dialect {
namespace ir {

OpNode TensorNode::producer() const {
  return OpNode(node_data_.dyn_cast<pir::OpResult>().owner());
}

OpNode TensorNode::ConsumerOpListView::Iterator::operator*() const {
  return OpNode(iter_.owner());
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
