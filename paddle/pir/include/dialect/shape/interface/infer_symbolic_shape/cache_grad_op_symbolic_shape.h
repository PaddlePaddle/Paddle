// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/pir/include/dialect/shape/utils/shape_analysis.h"

// The CacheGradOpSymbolicShapeInterface is used by forward operators to cache
// the symbolic shape information of the corresponding backward operators in the
// computational graph. With this interface, most backward operators do not need
// to implement the `InferSymbolicShapeInterface`.

namespace pir {

class CacheGradOpSymbolicShapeInterface
    : public pir::OpInterfaceBase<CacheGradOpSymbolicShapeInterface> {
 public:
  /// Defined these methods with the interface.
  struct Concept {
    explicit Concept(void (*cache_grad_op_symbolic_shape)(
        pir::Operation *op, pir::InferSymbolicShapeContext *infer_context))
        : cache_grad_op_symbolic_shape(cache_grad_op_symbolic_shape) {}
    void (*cache_grad_op_symbolic_shape)(
        pir::Operation *op, pir::InferSymbolicShapeContext *infer_context);
  };

  template <class ConcreteOp>
  struct Model : public Concept {
    static inline void CacheGradOpSymbolicShape(
        pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
      return op->dyn_cast<ConcreteOp>().CacheGradOpSymbolicShape(infer_context);
    }

    Model() : Concept(CacheGradOpSymbolicShape) {}
  };

  /// Constructor
  CacheGradOpSymbolicShapeInterface(const pir::Operation *op, Concept *impl)
      : pir::OpInterfaceBase<CacheGradOpSymbolicShapeInterface>(op),
        impl_(impl) {}

  void CacheGradOpSymbolicShape(pir::InferSymbolicShapeContext *infer_context);

 private:
  Concept *impl_;
};

}  // namespace pir

IR_DECLARE_EXPLICIT_TYPE_ID(pir::CacheGradOpSymbolicShapeInterface)
