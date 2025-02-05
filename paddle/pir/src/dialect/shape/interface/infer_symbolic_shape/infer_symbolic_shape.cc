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

#include "paddle/pir/include/dialect/shape/interface/infer_symbolic_shape/infer_symbolic_shape.h"

// This file implements the infer_symbolic_shape interface for both paddle and
// cinn operators.

// Add `interfaces : pir::InferSymbolicShapeInterface` in relative
// yaml file to corresponding op.

// Since necessary checks have been done in the Op's `InferMeta` and `VeriySig`,
// no more repetitive work here.

namespace pir {

bool InferSymbolicShapeInterface::InferSymbolicShape(
    pir::InferSymbolicShapeContext *infer_context) {
  return impl_->infer_symbolic_shapes(operation(), infer_context);
}

}  // namespace pir

IR_DEFINE_EXPLICIT_TYPE_ID(pir::InferSymbolicShapeInterface)
