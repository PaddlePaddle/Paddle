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

#include "paddle/ir/core/operation_utils.h"

namespace ir {
OperationArgument::OperationArgument(IrContext* ir_context, std::string name) {
  info_ = ir_context->GetRegisteredOpInfo(name);
}

OperationArgument::OperationArgument(OpInfo info,
                                     const std::vector<OpResult>& operands,
                                     const std::vector<Type>& types,
                                     const AttributeMap& named_attr)
    : info_(info),
      inputs_(operands),
      output_types_(types),
      attribute_(named_attr) {}

}  // namespace ir
