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

#include <memory>

#include "paddle/fluid/ir/dialect/pd_type.h"
#include "paddle/ir/core/operation.h"
#include "paddle/ir/core/value.h"

namespace ir {
namespace cinn {
namespace api {

class OpNode;

class TensorNode final {
 public:
  TensorNode(::ir::Value node_data) : node_data_(node_data) {}

  // Get the shape of tensor.
  const phi::DDim& shape() const {
    return node_data_.dyn_cast<paddle::dialect::DenseTensorType>().dims();
  }

  // Input data has no producer.
  bool HasProducer() const { return node_data_.begin() != node_data_.end(); }

  OpNode producer() const { return OpNode(node_data_.GetDefiningOp()); }

 private:
  ::ir::Value node_data_;
};

}  // namespace api
}  // namespace cinn
}  // namespace ir
