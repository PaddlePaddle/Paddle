// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/ir/core/builder.h"
#include "paddle/ir/core/region.h"

namespace ir {
Operation *Builder::insert(Operation *op) {
  if (block_) {
    block_->insert(insert_point_, op);
  } else {
    LOG(WARNING) << "Builder's Block is nullptr, insert failed.";
  }
  return op;
}

/// Create an operation given the fields represented as an OperationState.
Operation *Builder::create(OperationArgument &&argument) {
  return insert(Operation::create(std::move(argument)));
}

/// Creates an operation with the given fields.
Operation *Builder::create(const std::vector<ir::OpResult> &inputs,
                           const AttributeMap &attribute,
                           const std::vector<ir::Type> &output_types,
                           ir::OpInfo op_info) {
  return create(OperationArgument(inputs, attribute, output_types, op_info));
}

}  // namespace ir
