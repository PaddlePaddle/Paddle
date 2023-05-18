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

#include "paddle/fluid/dialect/pd_dialect.h"
#include "paddle/fluid/dialect/pd_type.h"
#include "paddle/fluid/dialect/utils.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/ir/builtin_type.h"
#include "paddle/ir/dialect_interface.h"
#include "paddle/phi/core/dense_tensor.h"

namespace paddle {
namespace dialect {
std::shared_ptr<paddle::framework::Variable>
ParameterConvertInterface::ParameterToVariable(ir::Parameter* parameter) {
  if (parameter->type().isa<DenseTensorType>()) {
    VLOG(4) << "Convert a DenseTensor Parameter to a variable.";
    std::shared_ptr<paddle::framework::Variable> var =
        std::make_shared<paddle::framework::Variable>();
    phi::DenseTensor* tensor = var->GetMutable<phi::DenseTensor>();
    // Init DenseTensor
    auto dim = parameter->type().dyn_cast<DenseTensorType>().dim();
    phi::DenseTensorMeta meta(
        TransToPhiDataType(
            parameter->type().dyn_cast<DenseTensorType>().dtype()),
        phi::DDim(dim.data(), dim.size()),
        TransToPhiDataLayout(
            parameter->type().dyn_cast<DenseTensorType>().data_layout()),
        parameter->type().dyn_cast<DenseTensorType>().lod(),
        parameter->type().dyn_cast<DenseTensorType>().offset());
    tensor->set_meta(meta);
    paddle::platform::DeviceContext* dev_ctx =
        paddle::platform::DeviceContextPool::Instance().Get(
            paddle::platform::CPUPlace());
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

std::unique_ptr<ir::Parameter> ParameterConvertInterface::VariableToParameter(
    paddle::framework::Variable* var) {
  if (var->IsType<phi::DenseTensor>()) {
    phi::DenseTensor* tensor = var->GetMutable<phi::DenseTensor>();
    // Get Meta
    ir::IrContext* ctx = ir::IrContext::Instance();
    ir::Type data_type = TransToIrDataType(tensor->dtype(), ctx);
    DenseTensorTypeStorage::Dim dims(tensor->dims().size());
    std::copy(tensor->dims().Get(),
              tensor->dims().Get() + tensor->dims().size(),
              dims.data());
    DenseTensorTypeStorage::DataLayout data_layout =
        TransToIrDataLayout(tensor->layout());
    DenseTensorTypeStorage::LoD lod = tensor->lod();
    size_t offset = tensor->meta().offset;
    void* data = tensor->data();
    ir::Type dense_tensor_type =
        DenseTensorType::get(ctx, data_type, dims, data_layout, lod, offset);
    return std::make_unique<ir::Parameter>(
        data,
        tensor->numel() * phi::SizeOf(tensor->dtype()),
        dense_tensor_type);
  } else {
    return nullptr;
  }
}

PaddleDialect::PaddleDialect(ir::IrContext* context)
    : ir::Dialect(name(), context, ir::TypeId::get<PaddleDialect>()) {
  initialize();
}

void PaddleDialect::initialize() {
  RegisterTypes<GET_PADDLE_TYPE_LIST>();
  RegisterInterfaces<ParameterConvertInterface>();
}

}  // namespace dialect
}  // namespace paddle
