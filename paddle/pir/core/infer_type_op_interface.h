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

// Type inference is currently modelled executionally for operation creation
// using the `InferMetaInterface`. While `InferShapedTypeOpInterface` is used to
// implement the shape and element type inference. The return type can often be
// deduced from the deduced return shape and elemental type (queryable from
// `InferShapedTypeOpInterface`) and so type inference for tensor types can be
// implemented with `InferShapedTypeOpInterface`.

namespace pir {

class InferShapedTypeOpInterface
    : public pir::OpInterfaceBase<InferShapedTypeOpInterface> {
 public:
  /// Defined these methods with the interface.
  struct Concept {
    explicit Concept(bool (*reify_return_type_shapes)(
        Operation* op,
        Builder& builder,  // NOLINT
        const std::vector<OpOperand>& operands,
        std::vector<Value>& reified_return_shapes))  // NOLINT
        : reify_return_type_shapes(reify_return_type_shapes) {}
    bool (*reify_return_type_shapes)(
        Operation* op,
        Builder& builder,
        const std::vector<OpOperand>& operands,
        std::vector<Value>& reified_return_shapes);  // NOLINT
  };

  template <class ConcreteOp>
  struct Model : public Concept {
    static inline bool ReifyReturnTypeShapes(
        Operation* op,
        Builder& builder,  // NOLINT
        const std::vector<OpOperand>& operands,
        std::vector<Value>& reified_return_shapes) {  // NOLINT
      return op->dyn_cast<ConcreteOp>().ReifyReturnTypeShapes(
          builder, operands, reified_return_shapes);
    }

    Model() : Concept(ReifyReturnTypeShapes) {}
  };

  /// Constructor
  InferShapedTypeOpInterface(Operation* op, Concept* impl)
      : pir::OpInterfaceBase<InferShapedTypeOpInterface>(op), impl_(impl) {}

  bool ReifyReturnTypeShapes(
      Builder& builder,  // NOLINT
      const std::vector<OpOperand>& operands,
      std::vector<Value>& reified_return_shapes);  // NOLINT

 private:
  Concept* impl_;
};

}  // namespace pir

IR_DECLARE_EXPLICIT_TYPE_ID(pir::InferShapedTypeOpInterface)
