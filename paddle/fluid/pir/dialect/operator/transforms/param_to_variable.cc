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

#include "paddle/fluid/pir/dialect/operator/transforms/param_to_variable.h"

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/dense_tensor.h"

namespace paddle::dialect {
std::shared_ptr<paddle::framework::Variable>
ParameterConvertInterface::ParameterToVariable(pir::Parameter *parameter) {
  if (parameter->type().isa<DenseTensorType>()) {
    VLOG(4) << "Convert a DenseTensor Parameter to a variable.";
    std::shared_ptr<paddle::framework::Variable> var =
        std::make_shared<paddle::framework::Variable>();
    phi::DenseTensor *tensor = var->GetMutable<phi::DenseTensor>();
    // Init DenseTensor
    auto dim = parameter->type().dyn_cast<DenseTensorType>().dims();
    phi::DenseTensorMeta meta(
        TransToPhiDataType(
            parameter->type().dyn_cast<DenseTensorType>().dtype()),
        dim,

        parameter->type().dyn_cast<DenseTensorType>().data_layout(),
        parameter->type().dyn_cast<DenseTensorType>().lod(),
        parameter->type().dyn_cast<DenseTensorType>().offset());
    tensor->set_meta(meta);
    phi::DeviceContext *dev_ctx =
        phi::DeviceContextPool::Instance().Get(phi::CPUPlace());
    dev_ctx->Alloc(tensor,
                   TransToPhiDataType(
                       parameter->type().dyn_cast<DenseTensorType>().dtype()));
    memcpy(tensor->data(),
           parameter->data(),
           tensor->numel() * phi::SizeOf(tensor->dtype()));
    return var;
  } else {
    return nullptr;
  }
}

std::unique_ptr<pir::Parameter> ParameterConvertInterface::VariableToParameter(
    paddle::framework::Variable *var) {
  if (var->IsType<phi::DenseTensor>()) {
    phi::DenseTensor *tensor = var->GetMutable<phi::DenseTensor>();
    // Get Meta
    pir::IrContext *ctx = pir::IrContext::Instance();
    pir::Type data_type = TransToIrDataType(tensor->dtype(), ctx);
    void *data = tensor->data();
    pir::Type dense_tensor_type = DenseTensorType::get(ctx,
                                                       data_type,
                                                       tensor->dims(),
                                                       tensor->layout(),
                                                       tensor->lod(),
                                                       tensor->meta().offset);
    return std::make_unique<pir::Parameter>(
        data,
        tensor->numel() * phi::SizeOf(tensor->dtype()),
        dense_tensor_type);
  } else {
    return nullptr;
  }
}

}  // namespace paddle::dialect

IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::ParameterConvertInterface)
