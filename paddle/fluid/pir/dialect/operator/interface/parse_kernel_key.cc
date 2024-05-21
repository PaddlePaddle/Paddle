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

#include "paddle/fluid/pir/dialect/operator/interface/parse_kernel_key.h"

namespace paddle::dialect {

KernelKeyTuple UniqueOpParseKernelKey(pir::Operation* op) {
  DenseTensorType x_type =
      op->operand_source(0).type().dyn_cast<paddle::dialect::DenseTensorType>();
  phi::DataType dtype = TransToPhiDataType(x_type.dtype());
  pir::BoolAttribute is_sort = op->attribute<pir::BoolAttribute>("is_sorted");
  phi::Backend backend = phi::Backend::UNDEFINED;
  if (is_sort.data()) {
    backend = phi::Backend::CPU;
  }
  return {dtype, backend};
}

KernelKeyTuple SaveCombineOpParseKernelKey(pir::Operation* op) {
  return {phi::DataType::FLOAT32, phi::Backend::UNDEFINED};
}

KernelKeyTuple NopOpParseKernelKey(pir::Operation* op) {
  return {phi::DataType::FLOAT32, phi::Backend::UNDEFINED};
}

KernelKeyTuple Nop_OpParseKernelKey(pir::Operation* op) {
  return {phi::DataType::FLOAT32, phi::Backend::UNDEFINED};
}

KernelKeyTuple PullGpupsSparseKernelKey(pir::Operation* op) {
  return {phi::DataType::FLOAT32, phi::Backend::UNDEFINED};
}

}  // namespace paddle::dialect

IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::ParseKernelKeyInterface)
