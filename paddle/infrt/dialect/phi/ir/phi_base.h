// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <string>

#include "paddle/infrt/dialect/phi/ir/infrt_phi_baseDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "paddle/infrt/dialect/phi/ir/infrt_phi_baseTypes.h.inc"

#define GET_OP_CLASSES
#include "paddle/infrt/dialect/phi/ir/infrt_phi_base.h.inc"

namespace mlir {
namespace OpTrait {

template <typename ConcreteType>
class PhiOpTrait : public OpTrait::TraitBase<ConcreteType, PhiOpTrait> {
 public:
  static LogicalResult verifyTrait(Operation *op) {
    return LogicalResult::success();
  }
};

}  // namespace OpTrait
}  // namespace mlir

namespace infrt {
namespace phi {}  // namespace phi
}  // namespace infrt
