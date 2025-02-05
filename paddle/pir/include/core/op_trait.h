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

#include "paddle/pir/include/core/dll_decl.h"
#include "paddle/pir/include/core/op_base.h"

namespace pir {

///
/// \brief Provides verification for ops that are known to have the
/// same operand shape.
///
class IR_API SameOperandsShapeTrait
    : public pir::OpTraitBase<SameOperandsShapeTrait> {
 public:
  explicit SameOperandsShapeTrait(const pir::Operation *op)
      : pir::OpTraitBase<SameOperandsShapeTrait>(op) {}
  static void Verify(Operation *op);
};

///
/// \brief Provides verification for ops that are known to have the
/// same operand and result shape.
///
class IR_API SameOperandsAndResultShapeTrait
    : public pir::OpTraitBase<SameOperandsAndResultShapeTrait> {
 public:
  explicit SameOperandsAndResultShapeTrait(const pir::Operation *op)
      : pir::OpTraitBase<SameOperandsAndResultShapeTrait>(op) {}
  static void Verify(Operation *op);
};

///
/// \brief Provides verification for ops that are known to have the
/// same operand element type (or the type itself if it is scalar).
///
class IR_API SameOperandsElementTypeTrait
    : public pir::OpTraitBase<SameOperandsElementTypeTrait> {
 public:
  explicit SameOperandsElementTypeTrait(const pir::Operation *op)
      : pir::OpTraitBase<SameOperandsElementTypeTrait>(op) {}
  static void Verify(Operation *op);
};

///
/// \brief Provides verification for ops that are known to have the
/// same operand and result element type (or the type itself if it is scalar).
///
class IR_API SameOperandsAndResultElementTypeTrait
    : public pir::OpTraitBase<SameOperandsAndResultElementTypeTrait> {
 public:
  explicit SameOperandsAndResultElementTypeTrait(const pir::Operation *op)
      : pir::OpTraitBase<SameOperandsAndResultElementTypeTrait>(op) {}
  static void Verify(Operation *op);
};

///
/// \brief Provides verification for ops that are known to have the
/// same operand and result type. It Subsumes both
/// SameOperandsAndResultShapeTrait and SameOperandsAndResultElementTypeTrait
///
class IR_API SameOperandsAndResultTypeTrait
    : public pir::OpTraitBase<SameOperandsAndResultTypeTrait> {
 public:
  explicit SameOperandsAndResultTypeTrait(const pir::Operation *op)
      : pir::OpTraitBase<SameOperandsAndResultTypeTrait>(op) {}

  static void Verify(Operation *op);
};

///
/// \brief Provides verification that all operands of the specified op have the
/// same type.
///
class IR_API SameTypeOperandsTrait
    : public pir::OpTraitBase<SameTypeOperandsTrait> {
 public:
  explicit SameTypeOperandsTrait(const pir::Operation *op)
      : pir::OpTraitBase<SameTypeOperandsTrait>(op) {}
  static void Verify(Operation *op);
};

///
/// \brief This trait provides return value APIs for ops that are known to have
/// a single result returned by GetType().
///
class IR_API OneResultTrait : public OpTraitBase<OneResultTrait> {
 public:
  using Base::Base;
  // Replace all uses of 'this' value with the new value, updating anything
  // in the IR that uses 'this' to use the other value instead.
  void ReplaceAllUsesWith(Value new_value) {
    this->operation()->result(0).ReplaceAllUsesWith(new_value);
  }

  // Replace all uses of 'this' value with the result of 'op'.
  void ReplaceAllUsesWith(Operation *op) {
    this->operation()->ReplaceAllUsesWith(op->result(0));
  }
  static void Verify(Operation *op);
};

///
/// \brief This trait marks the op can't be removed even if which has no output
/// or the output isn't used.
///
class IR_API SideEffectTrait : public OpTraitBase<SideEffectTrait> {
  using Base::Base;
};

///
/// \brief This trait marks the op's layout can't be modified.
///
class IR_API ImmutableLayoutTrait : public OpTraitBase<ImmutableLayoutTrait> {
 public:
  using Base::Base;
};

///
/// \brief This trait marks the op is elementwise and achieved
/// ElementwiseInferMeta.
///
class IR_API BinaryElementWiseTrait
    : public OpTraitBase<BinaryElementWiseTrait> {
 public:
  using Base::Base;
};

}  // namespace pir

IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::SameOperandsShapeTrait)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::SameOperandsAndResultShapeTrait)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::SameOperandsElementTypeTrait)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::SameOperandsAndResultElementTypeTrait)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::SameOperandsAndResultTypeTrait)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::SameTypeOperandsTrait)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::OneResultTrait)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::SideEffectTrait)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::ImmutableLayoutTrait)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::BinaryElementWiseTrait)
