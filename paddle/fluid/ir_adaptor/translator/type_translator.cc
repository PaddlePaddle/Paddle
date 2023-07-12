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
#include "paddle/fluid/ir/dialect/pd_type.h"
#include "paddle/fluid/ir/dialect/pd_type_storage.h"
#include "paddle/ir/core/builtin_type.h"

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
       [&](ir::IrContext* ctx, const VarDesc& var_desc) -> ir::Type {
         return ir::BoolType::get(ctx);
       }},
      {VarType::UINT8,
       [&](ir::IrContext* ctx, const VarDesc& var_desc) -> ir::Type {
         return ir::UInt8Type::get(ctx);
       }},
      {VarType::INT8,
       [&](ir::IrContext* ctx, const VarDesc& var_desc) -> ir::Type {
         return ir::Int8Type::get(ctx);
       }},
      {VarType::INT16,
       [&](ir::IrContext* ctx, const VarDesc& var_desc) -> ir::Type {
         return ir::Int16Type::get(ctx);
       }},
      {VarType::INT32,
       [&](ir::IrContext* ctx, const VarDesc& var_desc) -> ir::Type {
         return ir::Int32Type::get(ctx);
       }},
      {VarType::INT64,
       [&](ir::IrContext* ctx, const VarDesc& var_desc) -> ir::Type {
         return ir::Int64Type::get(ctx);
       }},
      {VarType::FP16,
       [&](ir::IrContext* ctx, const VarDesc& var_desc) -> ir::Type {
         return ir::Float16Type::get(ctx);
       }},
      {VarType::FP32,
       [&](ir::IrContext* ctx, const VarDesc& var_desc) -> ir::Type {
         return ir::Float32Type::get(ctx);
       }},
      {VarType::FP64,
       [&](ir::IrContext* ctx, const VarDesc& var_desc) -> ir::Type {
         return ir::Float64Type::get(ctx);
       }},
      {VarType::BF16,
       [&](ir::IrContext* ctx, const VarDesc& var_desc) -> ir::Type {
         return ir::BFloat16Type::get(ctx);
       }},
      {VarType::COMPLEX64,
       [&](ir::IrContext* ctx, const VarDesc& var_desc) -> ir::Type {
         return ir::Complex64Type::get(ctx);
       }},
      {VarType::COMPLEX128,
       [&](ir::IrContext* ctx, const VarDesc& var_desc) -> ir::Type {
         return ir::Complex128Type::get(ctx);
       }},
      {VarType::LOD_TENSOR,
       [&](ir::IrContext* ctx, const VarDesc& var_desc) -> ir::Type {
         VLOG(10) << "[vartype translating]"
                  << "[" << var_desc.Name() << "] from LOD_TENSOR";

         ir::Type dtype =
             this->operator[](var_desc.GetDataType())(ctx, var_desc);
         DenseTensorTypeStorage::Dim dim = phi::make_ddim(var_desc.GetShape());
         DenseTensorTypeStorage::DataLayout layout =
             DenseTensorTypeStorage::DataLayout::UNDEFINED;
         DenseTensorTypeStorage::LoD lod = {};
         size_t offset = 0;
         return DenseTensorType::get(ctx, dtype, dim, layout, lod, offset);
       }},
      {VarType::LOD_TENSOR_ARRAY,
       [&](ir::IrContext* ctx, const VarDesc& var_desc) -> ir::Type {
         VLOG(10) << "[vartype translating]"
                  << "[" << var_desc.Name() << "] from LOD_TENSOR_ARRAY";

         return ir::VectorType::get(ctx, std::vector<ir::Type>{});
       }},
      {VarType::SELECTED_ROWS,
       [&](ir::IrContext* ctx, const VarDesc& var_desc) -> ir::Type {
         VLOG(10) << "[vartype translating]"
                  << "[" << var_desc.Name() << "] from SELECTED_ROWS";

         ir::Type dtype =
             this->operator[](var_desc.GetDataType())(ctx, var_desc);

         SelectedRowsTypeStorage::Dim dim = phi::make_ddim(var_desc.GetShape());
         SelectedRowsTypeStorage::DataLayout layout =
             SelectedRowsTypeStorage::DataLayout::UNDEFINED;
         SelectedRowsTypeStorage::LoD lod = {};
         size_t offset = 0;
         ir::Type SelectedRows =
             SelectedRowsType::get(ctx, dtype, dim, layout, lod, offset);
         return SelectedRows;
       }},
  };
}

}  // namespace translator
}  // namespace paddle
