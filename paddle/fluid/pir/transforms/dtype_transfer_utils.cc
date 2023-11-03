// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pir/transforms/dtype_transfer_utils.h"
#include "paddle/fluid/pir/dialect/operator/interface/get_kernel_type_for_var.h"

#include "paddle/fluid/pir/dialect/kernel/ir/kernel_attribute.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_op.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/common/backend.h"
#include "paddle/phi/common/layout.h"
#include "paddle/pir/core/builtin_type.h"
#include "paddle/pir/core/op_info.h"

namespace pir {
phi::Kernel* GetKernel(pir::Operation* op, const phi::KernelKey& kernel_key) {
  auto& op_attributes = op->attributes();
  auto kernel_name =
      op_attributes.at("kernel_name").dyn_cast<pir::StrAttribute>().AsString();
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      kernel_name, kernel_key);
  auto phi_kernel = new phi::Kernel(kernel_result.kernel);
  return phi_kernel;
}

bool NeedTransformDataType(const phi::DataType& l, const phi::DataType& r) {
  return l != phi::DataType::ALL_DTYPE && r != phi::DataType::ALL_DTYPE &&
         l != r;
}

const phi::DataType GetKernelTypeforVar(
    pir::Operation* op,
    const std::string& var_name,
    const phi::DataType& tensor_dtype,
    const phi::KernelKey* expected_kernel_key) {
  //   auto phi_kernel = GetKernel(op, *expected_kernel_key);

  //   bool has_infer_varkernel_fn =
  //       phi_kernel && phi_kernel->get_kerneltype_forvar_fn_ != nullptr;

  pir::IrContext* ctx = pir::IrContext::Instance();
  pir::OpInfo op_info = ctx->GetRegisteredOpInfo(op->name());

  auto get_kernel_type_for_var_interface =
      op_info.GetInterfaceImpl<paddle::dialect::GetKernelTypeForVarInterface>();
  PADDLE_ENFORCE_NOT_NULL(
      get_kernel_type_for_var_interface,
      phi::errors::NotFound(
          "can not find GetKernelTypeForVarInterface from [%s]", op->name()));

  phi::DataType kernel_dtype_for_var =
      get_kernel_type_for_var_interface->get_kernel_type_for_var_(
          var_name, tensor_dtype, (*expected_kernel_key).dtype());

  return kernel_dtype_for_var;
}

pir::Type BuildDtypeTransferOutputType(pir::Type type,
                                       const phi::Place& place,
                                       phi::DataType data_dtype,
                                       pir::IrContext* ctx) {
  if (type.isa<paddle::dialect::AllocatedDenseTensorType>()) {
    auto dense_tensor_type =
        type.dyn_cast<paddle::dialect::AllocatedDenseTensorType>();

    auto out_dtype = paddle::dialect::TransToIrDataType(data_dtype, ctx);
    return paddle::dialect::AllocatedDenseTensorType::get(
        ctx,
        place,
        out_dtype,
        dense_tensor_type.dims(),
        dense_tensor_type.data_layout(),
        dense_tensor_type.lod(),
        dense_tensor_type.offset());

  } else if (type.isa<paddle::dialect::AllocatedSelectedRowsType>()) {
    auto selected_rows_type =
        type.dyn_cast<paddle::dialect::AllocatedSelectedRowsType>();
    auto out_dtype = paddle::dialect::TransToIrDataType(data_dtype, ctx);
    return paddle::dialect::AllocatedSelectedRowsType::get(
        ctx,
        place,
        out_dtype,
        selected_rows_type.dims(),
        selected_rows_type.data_layout(),
        selected_rows_type.lod(),
        selected_rows_type.offset());
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "BuildOutputType only support DenseTensorType and SelectedRowsType"));
  }
}

pir::OpResult AddDtypeTransferOp(pir::Value in,
                                 pir::Block* block,
                                 const phi::KernelKey& kernel_key,
                                 const phi::Place& out_place,
                                 const phi::DataType& src_dtype,
                                 const phi::DataType& dst_dtype) {
  pir::IrContext* ctx = pir::IrContext::Instance();

  pir::OpInfo phi_kernel_op_info =
      ctx->GetRegisteredOpInfo(paddle::dialect::PhiKernelOp::name());

  // Get kernelkey (backend、layout)
  phi::Backend kernel_backend = phi::Backend::UNDEFINED;
  phi::DataLayout kernel_layout = phi::DataLayout::UNDEFINED;
  if (in.type().isa<paddle::dialect::AllocatedDenseTensorType>()) {
    kernel_backend = paddle::experimental::ParseBackend(
        in.type()
            .dyn_cast<paddle::dialect::AllocatedDenseTensorType>()
            .place());
    kernel_layout = paddle::experimental::ParseLayout(
        in.type()
            .dyn_cast<paddle::dialect::AllocatedDenseTensorType>()
            .data_layout());
  } else if (in.type().isa<paddle::dialect::AllocatedSelectedRowsType>()) {
    kernel_backend = paddle::experimental::ParseBackend(
        in.type()
            .dyn_cast<paddle::dialect::AllocatedSelectedRowsType>()
            .place());
    kernel_layout = paddle::experimental::ParseLayout(
        in.type()
            .dyn_cast<paddle::dialect::AllocatedSelectedRowsType>()
            .data_layout());
  } else {
    PADDLE_THROW(
        phi::errors::Unimplemented("Get kernelkey for CastOp only support "
                                   "DenseTensorType and SelectedRowsType"));
  }
  phi::KernelKey cast_kernel_key(kernel_backend, kernel_layout, src_dtype);

  // Create CastOp
  std::unordered_map<std::string, pir::Attribute> op_attribute{
      {"op_name", pir::StrAttribute::get(ctx, "pd_op.cast")},
      {"kernel_name", pir::StrAttribute::get(ctx, "cast")},
      {"kernel_key",
       paddle::dialect::KernelAttribute::get(ctx, cast_kernel_key)},
      {"dtype", paddle::dialect::DataTypeAttribute::get(ctx, dst_dtype)}};

  pir::Type output_types =
      BuildDtypeTransferOutputType(in.type(), out_place, dst_dtype, ctx);

  pir::Operation* op = pir::Operation::Create(
      {in}, op_attribute, {output_types}, phi_kernel_op_info);

  auto in_op = in.dyn_cast<pir::OpResult>().owner();
  if (in_op && in_op->HasAttribute(kAttrIsPersisable)) {
    op->set_attribute(kAttrIsPersisable, in_op->attribute(kAttrIsPersisable));
  }
  block->push_back(op);
  pir::OpResult new_in = op->result(0);
  return new_in;
}

}  // namespace pir
