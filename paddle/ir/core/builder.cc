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
#include "paddle/ir/core/value.h"

namespace ir {
/// Create an operation given the fields represented as an OperationState.
Operation *Builder::Build(OperationArgument &&argument) {
  return Insert(Operation::Create(std::move(argument)));
}

/// Creates an operation with the given fields.
Operation *Builder::Build(const std::vector<OpResult> &inputs,
                          const AttributeMap &attribute,
                          const std::vector<Type> &output_types,
                          OpInfo op_info) {
  return Build(OperationArgument(inputs, attribute, output_types, op_info));
}

Operation *Builder::Insert(Operation *op) {
  if (block_) {
    block_->insert(insert_point_, op);
  } else {
    LOG(WARNING) << "Builder's Block is nullptr, insert failed.";
  }
  return op;
}

}  // namespace ir
