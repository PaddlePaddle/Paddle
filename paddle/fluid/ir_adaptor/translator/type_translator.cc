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

#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/type_storage.h"
#include "paddle/phi/core/framework/framework.pb.h"
#include "paddle/pir/include/core/builtin_type.h"

namespace paddle::translator {

using OpDesc = paddle::framework::OpDesc;
using BlockDesc = paddle::framework::BlockDesc;
using VarDesc = paddle::framework::VarDesc;
using VarType = paddle::framework::proto::VarType;
using DenseTensorType = paddle::dialect::DenseTensorType;
using DenseTensorTypeStorage = paddle::dialect::DenseTensorTypeStorage;
using SelectedRowsType = paddle::dialect::SelectedRowsType;
using SelectedRowsTypeStorage = paddle::dialect::SelectedRowsTypeStorage;
using SparseCooTensorType = paddle::dialect::SparseCooTensorType;
using SparseCooTensorTypeStorage = paddle::dialect::SparseCooTensorTypeStorage;
using SparseCsrTensorType = paddle::dialect::SparseCsrTensorType;
using SparseCsrTensorTypeStorage = paddle::dialect::SparseCsrTensorTypeStorage;
using DataLayout = DenseTensorTypeStorage::DataLayout;
using LegacyLoD = DenseTensorTypeStorage::LegacyLoD;

TypeTranslator::TypeTranslator() {
  const auto& HandleTensor = [&](pir::IrContext* ctx,
                                 const VarDesc& var_desc) -> pir::Type {
    VLOG(10) << "[vartype translating]"
             << "[" << var_desc.Name() << "] from DENSE_TENSOR";
    const pir::Type dtype =
        this->operator[](var_desc.GetDataType())(ctx, var_desc);
    const auto dim = common::make_ddim(var_desc.GetShape());
    const auto layout = DataLayout::NCHW;
    const LegacyLoD lod = {};
    const size_t offset = 0;
    return DenseTensorType::get(ctx, dtype, dim, layout, lod, offset);
  };
  const auto& HandleTensorArray = [&](pir::IrContext* ctx,
                                      const VarDesc& var_desc) -> pir::Type {
    VLOG(10) << "[vartype translating]"
             << "[" << var_desc.Name() << "] from DENSE_TENSOR_ARRAY";
    const pir::Type dtype =
        this->operator[](var_desc.GetDataType())(ctx, var_desc);
    const auto dims = common::make_ddim(var_desc.GetShape());
    const auto layout = DataLayout::NCHW;
    return paddle::dialect::DenseTensorArrayType::get(ctx, dtype, dims, layout);
  };

  const auto& HandleSelectedRows = [&](pir::IrContext* ctx,
                                       const VarDesc& var_desc) -> pir::Type {
    VLOG(10) << "[vartype translating]"
             << "[" << var_desc.Name() << "] from SELECTED_ROWS";
    const pir::Type dtype =
        this->operator[](var_desc.GetDataType())(ctx, var_desc);
    const auto dim = common::make_ddim(var_desc.GetShape());
    const auto layout = DataLayout::NCHW;
    const LegacyLoD lod = {};
    const size_t offset = 0;
    pir::Type SelectedRows =
        SelectedRowsType::get(ctx, dtype, dim, layout, lod, offset);
    return SelectedRows;
  };

  const auto& HandleSparseCooTensor =
      [&](pir::IrContext* ctx, const VarDesc& var_desc) -> pir::Type {
    VLOG(10) << "[vartype translating]"
             << "[" << var_desc.Name() << "] from SPARSE_COO";
    const pir::Type dtype =
        this->operator[](var_desc.GetDataType())(ctx, var_desc);
    const auto dim = common::make_ddim(var_desc.GetShape());
    const LegacyLoD lod = {};
    const size_t offset = 0;
    bool coalesced = false;
    const auto non_zero_dims = common::make_ddim(var_desc.GetShape());
    const auto layout = DataLayout::NCHW;
    pir::DenseTensorType non_zero_indices =
        DenseTensorType::get(ctx, dtype, dim, layout, lod, offset);
    pir::DenseTensorType non_zero_elements =
        DenseTensorType::get(ctx, dtype, dim, layout, lod, offset);
    pir::Type SparseCooTensor = SparseCooTensorType::get(ctx,
                                                         dtype,
                                                         dim,
                                                         non_zero_dims,
                                                         layout,
                                                         non_zero_indices,
                                                         non_zero_elements,
                                                         coalesced);
    return SparseCooTensor;
  };

  const auto& HandleSparseCsrTensor =
      [&](pir::IrContext* ctx, const VarDesc& var_desc) -> pir::Type {
    VLOG(10) << "[vartype translating]"
             << "[" << var_desc.Name() << "] from SPARSE_CSR";
    const pir::Type dtype =
        this->operator[](var_desc.GetDataType())(ctx, var_desc);
    const auto dim = common::make_ddim(var_desc.GetShape());
    const LegacyLoD lod = {};
    const size_t offset = 0;
    const auto layout = DataLayout::NCHW;
    pir::DenseTensorType non_zero_crows =
        DenseTensorType::get(ctx, dtype, dim, layout, lod, offset);
    pir::DenseTensorType non_zero_cols =
        DenseTensorType::get(ctx, dtype, dim, layout, lod, offset);
    pir::DenseTensorType non_zero_elements =
        DenseTensorType::get(ctx, dtype, dim, layout, lod, offset);
    pir::Type SparseCsrTensor = SparseCsrTensorType::get(ctx,
                                                         dtype,
                                                         dim,
                                                         layout,
                                                         non_zero_crows,
                                                         non_zero_cols,
                                                         non_zero_elements);
    return SparseCsrTensor;
  };

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
      {VarType::FP8_E4M3FN,
       [&](pir::IrContext* ctx, const VarDesc& var_desc) -> pir::Type {
         return pir::Float8E4M3FNType::get(ctx);
       }},
      {VarType::FP8_E5M2,
       [&](pir::IrContext* ctx, const VarDesc& var_desc) -> pir::Type {
         return pir::Float8E5M2Type::get(ctx);
       }},
      {VarType::DENSE_TENSOR, HandleTensor},
      {VarType::DENSE_TENSOR_ARRAY, HandleTensorArray},
      {VarType::SELECTED_ROWS, HandleSelectedRows},
      {VarType::SPARSE_COO, HandleSparseCooTensor},
      {VarType::SPARSE_CSR, HandleSparseCsrTensor},
  };
}

}  // namespace paddle::translator
