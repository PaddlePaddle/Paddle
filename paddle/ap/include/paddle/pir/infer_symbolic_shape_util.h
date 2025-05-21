// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/common/ddim.h"
#include "paddle/common/layout.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/pir/include/dialect/shape/utils/shape_analysis.h"

namespace ap::dialect {

// for ap_op.facade
bool ApOpFacadeOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context);

// for pd_op.ap_facade
bool PdOpApFacadeOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context);

// for pd_op.ap_variadic
bool PdOpApVariadicOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context);

}  // namespace ap::dialect
