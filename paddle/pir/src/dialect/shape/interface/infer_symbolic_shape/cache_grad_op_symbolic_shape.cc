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

#include "paddle/pir/include/dialect/shape/interface/infer_symbolic_shape/cache_grad_op_symbolic_shape.h"

// This file implements the cache_grad_op_symbolic_shape interface for both
// paddle and cinn operators.

// Add `interfaces : pir::CacheGradOpSymbolicShapeInterface` in relative
// yaml file to corresponding op.

namespace pir {

bool CacheGradOpSymbolicShapeInterface::CacheGradOpSymbolicShape(
    pir::InferSymbolicShapeContext *infer_context) {
  return impl_->cache_grad_op_symbolic_shape(operation(), infer_context);
}

}  // namespace pir

IR_DEFINE_EXPLICIT_TYPE_ID(pir::CacheGradOpSymbolicShapeInterface)
