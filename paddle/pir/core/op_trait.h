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

#pragma once

#include "paddle/pir/core/op_base.h"

namespace pir::op_trait {

namespace impl {
void VerifySameOperandsShapeTrait(Operation *op);
void VerifySameOperandsAndResultShapeTrait(Operation *op);
void VerifySameOperandsElementTypeTrait(Operation *op);
void VerifySameOperandsAndResultElementTypeTrait(Operation *op);
void VerifySameOperandsAndResultTypeTrait(Operation *op);
void VerifySameTypeOperandsTrait(Operation *op);
}  // namespace impl

///
/// \brief Provides verification for ops that are known to have the
/// same operand shape.
///
class SameOperandsShapeTrait : public pir::OpTraitBase<SameOperandsShapeTrait> {
 public:
  explicit SameOperandsShapeTrait(pir::Operation *op)
      : pir::OpTraitBase<SameOperandsShapeTrait>(op) {}
  static void Verify(Operation *op) {
    return impl::VerifySameOperandsShapeTrait(op);
  }
};

///
/// \brief Provides verification for ops that are known to have the
/// same operand and result shape.
///
class SameOperandsAndResultShapeTrait
    : public pir::OpTraitBase<SameOperandsAndResultShapeTrait> {
 public:
  explicit SameOperandsAndResultShapeTrait(pir::Operation *op)
      : pir::OpTraitBase<SameOperandsAndResultShapeTrait>(op) {}
  static void Verify(Operation *op) {
    return impl::VerifySameOperandsAndResultShapeTrait(op);
  }
};

///
/// \brief Provides verification for ops that are known to have the
/// same operand element type (or the type itself if it is scalar).
///
class SameOperandsElementTypeTrait
    : public pir::OpTraitBase<SameOperandsElementTypeTrait> {
 public:
  explicit SameOperandsElementTypeTrait(pir::Operation *op)
      : pir::OpTraitBase<SameOperandsElementTypeTrait>(op) {}
  static void Verify(Operation *op) {
    return impl::VerifySameOperandsElementTypeTrait(op);
  }
};

///
/// \brief Provides verification for ops that are known to have the
/// same operand and result element type (or the type itself if it is scalar).
///
class SameOperandsAndResultElementTypeTrait
    : public pir::OpTraitBase<SameOperandsAndResultElementTypeTrait> {
 public:
  explicit SameOperandsAndResultElementTypeTrait(pir::Operation *op)
      : pir::OpTraitBase<SameOperandsAndResultElementTypeTrait>(op) {}
  static void Verify(Operation *op) {
    return impl::VerifySameOperandsAndResultElementTypeTrait(op);
  }
};

///
/// \brief Provides verification for ops that are known to have the
/// same operand and result type. It Subsumes both
/// SameOperandsAndResultShapeTrait and SameOperandsAndResultElementTypeTrait
///
class SameOperandsAndResultTypeTrait
    : public pir::OpTraitBase<SameOperandsAndResultTypeTrait> {
 public:
  explicit SameOperandsAndResultTypeTrait(pir::Operation *op)
      : pir::OpTraitBase<SameOperandsAndResultTypeTrait>(op) {}

  static void Verify(Operation *op) {
    return impl::VerifySameOperandsAndResultTypeTrait(op);
  }
};

///
/// \brief Provides verification that all operands of the specified op have the
/// same type.
///
class SameTypeOperandsTrait : public pir::OpTraitBase<SameTypeOperandsTrait> {
 public:
  explicit SameTypeOperandsTrait(pir::Operation *op)
      : pir::OpTraitBase<SameTypeOperandsTrait>(op) {}
  static void Verify(Operation *op) {
    return impl::VerifySameTypeOperandsTrait(op);
  }
};

}  // namespace pir::op_trait

IR_DECLARE_EXPLICIT_TYPE_ID(pir::op_trait::SameOperandsShapeTrait)
IR_DECLARE_EXPLICIT_TYPE_ID(pir::op_trait::SameOperandsAndResultShapeTrait)
IR_DECLARE_EXPLICIT_TYPE_ID(pir::op_trait::SameOperandsElementTypeTrait)
IR_DECLARE_EXPLICIT_TYPE_ID(
    pir::op_trait::SameOperandsAndResultElementTypeTrait)
IR_DECLARE_EXPLICIT_TYPE_ID(pir::op_trait::SameOperandsAndResultTypeTrait)
IR_DECLARE_EXPLICIT_TYPE_ID(pir::op_trait::SameTypeOperandsTrait)
