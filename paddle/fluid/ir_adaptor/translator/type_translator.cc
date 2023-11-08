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

#include "paddle/fluid/ir_adaptor/translator/type_translator.h"

#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/type_storage.h"
#include "paddle/pir/core/builtin_type.h"

namespace paddle {
namespace translator {

using OpDesc = paddle::framework::OpDesc;
using BlockDesc = paddle::framework::BlockDesc;
using VarDesc = paddle::framework::VarDesc;
using VarType = paddle::framework::proto::VarType;
using DenseTensorType = paddle::dialect::DenseTensorType;
using DenseTensorTypeStorage = paddle::dialect::DenseTensorTypeStorage;
using SelectedRowsType = paddle::dialect::SelectedRowsType;
using SelectedRowsTypeStorage = paddle::dialect::SelectedRowsTypeStorage;

TypeTranslator::TypeTranslator() {
  handlers = {
      {VarType::BOOL,
       [&](pir::IrContext* ctx, const VarDesc& var_desc) -> pir::Type {
         return pir::BoolType::get(ctx);
       }},
      {VarType::UINT8,
       [&](pir::IrContext* ctx, const VarDesc& var_desc) -> pir::Type {
         return pir::UInt8Type::get(ctx);
       }},
      {VarType::INT8,
       [&](pir::IrContext* ctx, const VarDesc& var_desc) -> pir::Type {
         return pir::Int8Type::get(ctx);
       }},
      {VarType::INT16,
       [&](pir::IrContext* ctx, const VarDesc& var_desc) -> pir::Type {
         return pir::Int16Type::get(ctx);
       }},
      {VarType::INT32,
       [&](pir::IrContext* ctx, const VarDesc& var_desc) -> pir::Type {
         return pir::Int32Type::get(ctx);
       }},
      {VarType::INT64,
       [&](pir::IrContext* ctx, const VarDesc& var_desc) -> pir::Type {
         return pir::Int64Type::get(ctx);
       }},
      {VarType::FP16,
       [&](pir::IrContext* ctx, const VarDesc& var_desc) -> pir::Type {
         return pir::Float16Type::get(ctx);
       }},
      {VarType::FP32,
       [&](pir::IrContext* ctx, const VarDesc& var_desc) -> pir::Type {
         return pir::Float32Type::get(ctx);
       }},
      {VarType::FP64,
       [&](pir::IrContext* ctx, const VarDesc& var_desc) -> pir::Type {
         return pir::Float64Type::get(ctx);
       }},
      {VarType::BF16,
       [&](pir::IrContext* ctx, const VarDesc& var_desc) -> pir::Type {
         return pir::BFloat16Type::get(ctx);
       }},
      {VarType::COMPLEX64,
       [&](pir::IrContext* ctx, const VarDesc& var_desc) -> pir::Type {
         return pir::Complex64Type::get(ctx);
       }},
      {VarType::COMPLEX128,
       [&](pir::IrContext* ctx, const VarDesc& var_desc) -> pir::Type {
         return pir::Complex128Type::get(ctx);
       }},
      {VarType::LOD_TENSOR,
       [&](pir::IrContext* ctx, const VarDesc& var_desc) -> pir::Type {
         VLOG(10) << "[vartype translating]"
                  << "[" << var_desc.Name() << "] from LOD_TENSOR";

         pir::Type dtype =
             this->operator[](var_desc.GetDataType())(ctx, var_desc);
         DenseTensorTypeStorage::Dim dim = phi::make_ddim(var_desc.GetShape());
         DenseTensorTypeStorage::DataLayout layout =
             DenseTensorTypeStorage::DataLayout::UNDEFINED;
         DenseTensorTypeStorage::LoD lod = {};
         size_t offset = 0;
         return DenseTensorType::get(ctx, dtype, dim, layout, lod, offset);
       }},
      {VarType::LOD_TENSOR_ARRAY,
       [&](pir::IrContext* ctx, const VarDesc& var_desc) -> pir::Type {
         VLOG(10) << "[vartype translating]"
                  << "[" << var_desc.Name() << "] from LOD_TENSOR_ARRAY";

         return pir::VectorType::get(ctx, std::vector<pir::Type>{});
       }},
      {VarType::SELECTED_ROWS,
       [&](pir::IrContext* ctx, const VarDesc& var_desc) -> pir::Type {
         VLOG(10) << "[vartype translating]"
                  << "[" << var_desc.Name() << "] from SELECTED_ROWS";

         pir::Type dtype =
             this->operator[](var_desc.GetDataType())(ctx, var_desc);

         SelectedRowsTypeStorage::Dim dim = phi::make_ddim(var_desc.GetShape());
         SelectedRowsTypeStorage::DataLayout layout =
             SelectedRowsTypeStorage::DataLayout::UNDEFINED;
         SelectedRowsTypeStorage::LoD lod = {};
         size_t offset = 0;
         pir::Type SelectedRows =
             SelectedRowsType::get(ctx, dtype, dim, layout, lod, offset);
         return SelectedRows;
       }},
  };
}

}  // namespace translator
}  // namespace paddle
