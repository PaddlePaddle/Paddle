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

#include "paddle/fluid/pir/dialect/operator/interface/decomp.h"
#include "paddle/fluid/pir/dialect/operator/interface/get_kernel_type_for_var.h"
#include "paddle/fluid/pir/dialect/operator/interface/infermeta.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/interface/parse_kernel_key.h"
#include "paddle/fluid/pir/dialect/operator/interface/vjp.h"

namespace paddle {
namespace dialect {
std::vector<std::vector<pir::OpResult>> VjpInterface::Vjp(
    pir::Operation* op,
    const std::vector<std::vector<pir::Value>>& inputs,
    const std::vector<std::vector<pir::OpResult>>& outputs,
    const std::vector<std::vector<pir::OpResult>>& out_grads,
    const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<pir::Value>> out_grads_value;
  for (const auto& grad : out_grads) {
    std::vector<pir::Value> grad_value;
    for (auto op_result : grad) {
      grad_value.emplace_back(op_result);
    }
    out_grads_value.emplace_back(std::move(grad_value));
  }
  return impl_->vjp_(op, inputs, outputs, out_grads_value, stop_gradients);
}

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

}  // namespace dialect
}  // namespace paddle

IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::InferMetaInterface)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::OpYamlInfoInterface)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::VjpInterface)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::DecompInterface)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::GetKernelTypeForVarInterface)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::ParseKernelKeyInterface)
