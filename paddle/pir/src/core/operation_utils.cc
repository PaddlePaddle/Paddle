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

#include "paddle/pir/include/core/operation_utils.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/region.h"

namespace pir {
OperationArgument::OperationArgument(IrContext* ir_context,
                                     const std::string& name) {
  info = ir_context->GetRegisteredOpInfo(name);
}

Region& OperationArgument::AddRegion() {
  regions.emplace_back(new Region);
  return *regions.back();
}

/// Take a region that should be attached to the Operation.
void OperationArgument::AddRegion(std::unique_ptr<Region>&& region) {
  regions.emplace_back(std::move(region));
}

void OperationArgument::AddRegions(size_t size) {
  for (size_t i = 0; i < size; ++i) {
    regions.emplace_back(nullptr);
  }
}

}  // namespace pir
