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

#include "paddle/fluid/ir/dialect/pd_dialect.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/ir/dialect/pd_attribute.h"
// NOTE(zhangbo9674): File pd_op.h is generated by op_gen.py, see details in
// paddle/fluid/ir/dialect/CMakeLists.txt.
#include "paddle/fluid/ir/dialect/pd_op.h"
#include "paddle/fluid/ir/dialect/pd_type.h"
#include "paddle/fluid/ir/dialect/pd_type_storage.h"
#include "paddle/fluid/ir/dialect/utils.h"
#include "paddle/ir/core/dialect_interface.h"
#include "paddle/ir/core/utils.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/dense_tensor.h"

namespace paddle {
namespace dialect {
std::shared_ptr<paddle::framework::Variable>
ParameterConvertInterface::ParameterToVariable(ir::Parameter *parameter) {
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
    paddle::platform::DeviceContext *dev_ctx =
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
    paddle::framework::Variable *var) {
  if (var->IsType<phi::DenseTensor>()) {
    phi::DenseTensor *tensor = var->GetMutable<phi::DenseTensor>();
    // Get Meta
    ir::IrContext *ctx = ir::IrContext::Instance();
    ir::Type data_type = TransToIrDataType(tensor->dtype(), ctx);
    void *data = tensor->data();
    ir::Type dense_tensor_type = DenseTensorType::get(ctx,
                                                      data_type,
                                                      tensor->dims(),
                                                      tensor->layout(),
                                                      tensor->lod(),
                                                      tensor->meta().offset);
    return std::make_unique<ir::Parameter>(
        data,
        tensor->numel() * phi::SizeOf(tensor->dtype()),
        dense_tensor_type);
  } else {
    return nullptr;
  }
}

PaddleDialect::PaddleDialect(ir::IrContext *context)
    : ir::Dialect(name(), context, ir::TypeId::get<PaddleDialect>()) {
  initialize();
}

void PaddleDialect::initialize() {
  RegisterTypes<paddle::dialect::DenseTensorType>();
  RegisterTypes<paddle::dialect::SelectedRowsType>();

  RegisterAttributes<paddle::dialect::IntArrayAttribute,
                     paddle::dialect::DataTypeAttribute,
                     paddle::dialect::PlaceAttribute,
                     paddle::dialect::DataLayoutAttribute>();

  // NOTE(zhangbo9674): GET_OP_LIST is defined in pd_op.h which is
  // generated by op_gen.py, see details in
  // paddle/fluid/ir/dialect/CMakeLists.txt.
  // NOTE(Ruting)GET_MANUAL_OP_LIST is define in pd_manual_op.h"
  // use RegisterOps when list has more than two ops.
  RegisterOps<
#define GET_OP_LIST
#include "paddle/fluid/ir/dialect/pd_op.h"  // NOLINT
      >();
  RegisterOp<paddle::dialect::AddNOp>();

  RegisterInterfaces<ParameterConvertInterface>();
}

void PaddleDialect::PrintType(ir::Type type, std::ostream &os) const {
  os << type.dialect().name();
  os << '.';
  if (auto tensor_type = type.dyn_cast<DenseTensorType>()) {
    os << "tensor<";
    for (auto d : phi::vectorize(tensor_type.dims())) {
      os << d;
      os << "x";
    }
    tensor_type.dtype().Print(os);
    os << ">";
  } else if (auto selected_rows_type = type.dyn_cast<SelectedRowsType>()) {
    os << "selectedrows<";
    for (auto d : phi::vectorize(selected_rows_type.dims())) {
      os << d;
      os << "x";
    }
    selected_rows_type.dtype().Print(os);
    os << ">";
  }
}

void PaddleDialect::PrintAttribute(ir::Attribute attr, std::ostream &os) const {
  if (auto int_array_attr = attr.dyn_cast<IntArrayAttribute>()) {
    phi::IntArray data = int_array_attr.data();
    os << "IntArray[";
    const auto &inner_data = data.GetData();
    ir::PrintInterleave(
        inner_data.begin(),
        inner_data.end(),
        [&os](int64_t i) { os << i; },
        [&os]() { os << ","; });
    os << "]";
  } else if (auto data_type_attr = attr.dyn_cast<DataTypeAttribute>()) {
    os << data_type_attr.data();
  } else if (auto place_type_attr = attr.dyn_cast<PlaceAttribute>()) {
    os << place_type_attr.data();
  } else if (auto data_layout_attr = attr.dyn_cast<DataLayoutAttribute>()) {
    os << data_layout_attr.data();
  } else {
    os << "<#AttrNotImplemented>";
  }
}

}  // namespace dialect
}  // namespace paddle

IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::PaddleDialect)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::ParameterConvertInterface)
