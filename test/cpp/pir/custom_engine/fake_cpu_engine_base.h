// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include <unordered_map>

#include "paddle/fluid/pir/dialect/kernel/ir/kernel_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/pir/include/core/block.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/operation.h"

using paddle::dialect::AllocatedDenseTensorArrayType;
using paddle::dialect::AllocatedDenseTensorType;
using paddle::dialect::AllocatedSelectedRowsType;
using paddle::dialect::AllocatedSparseCooTensorType;
using paddle::dialect::AllocatedSparseCsrTensorType;
using paddle::dialect::DenseTensorArrayType;
using paddle::dialect::DenseTensorType;
using paddle::dialect::SelectedRowsType;
using paddle::dialect::SparseCooTensorType;
using paddle::dialect::SparseCsrTensorType;

template <class IrType1, class IrType2>
static pir::Type create_sparse_coo_tensor_type(pir::Type type,
                                               const phi::Place& place,
                                               pir::Type out_dtype,
                                               pir::IrContext* ctx) {
  auto input_type = type.dyn_cast<IrType1>();
  return IrType2::get(ctx,
                      place,
                      out_dtype,
                      input_type.dims(),
                      input_type.non_zero_dims(),
                      input_type.data_layout(),
                      input_type.non_zero_indices(),
                      input_type.non_zero_elements(),
                      input_type.coalesced());
}

template <class IrType1, class IrType2>
static pir::Type create_sparse_csr_tensor_type(pir::Type type,
                                               const phi::Place& place,
                                               pir::Type out_dtype,
                                               pir::IrContext* ctx) {
  auto input_type = type.dyn_cast<IrType1>();
  return IrType2::get(ctx,
                      place,
                      out_dtype,
                      input_type.dims(),
                      input_type.data_layout(),
                      input_type.non_zero_crows(),
                      input_type.non_zero_cols(),
                      input_type.non_zero_elements());
}

template <class IrType1, class IrType2>
static pir::Type create_type(pir::Type type,
                             const phi::Place& place,
                             pir::Type out_dtype,
                             pir::IrContext* ctx) {
  auto input_type = type.dyn_cast<IrType1>();
  return IrType2::get(ctx,
                      place,
                      out_dtype,
                      input_type.dims(),
                      input_type.data_layout(),
                      input_type.lod(),
                      input_type.offset());
}

static pir::Type BuildOutputType(pir::Type type,
                                 const phi::Place& place,
                                 pir::IrContext* ctx) {
  if (type.isa<DenseTensorType>()) {
    auto out_dtype = type.dyn_cast<DenseTensorType>().dtype();
    return create_type<DenseTensorType, AllocatedDenseTensorType>(
        type, place, out_dtype, ctx);
  } else if (type.isa<SelectedRowsType>()) {
    auto out_dtype = type.dyn_cast<SelectedRowsType>().dtype();
    return create_type<SelectedRowsType, AllocatedSelectedRowsType>(
        type, place, out_dtype, ctx);
  } else if (type.isa<DenseTensorArrayType>()) {
    auto array_type = type.dyn_cast<DenseTensorArrayType>();
    return AllocatedDenseTensorArrayType::get(ctx,
                                              place,
                                              array_type.dtype(),
                                              array_type.dims(),
                                              array_type.data_layout());
  } else if (type.isa<SparseCooTensorType>()) {
    auto out_dtype = type.dyn_cast<SparseCooTensorType>().dtype();
    return create_sparse_coo_tensor_type<SparseCooTensorType,
                                         AllocatedSparseCooTensorType>(
        type, place, out_dtype, ctx);
  } else if (type.isa<SparseCsrTensorType>()) {
    auto out_dtype = type.dyn_cast<SparseCsrTensorType>().dtype();
    return create_sparse_csr_tensor_type<SparseCsrTensorType,
                                         AllocatedSparseCsrTensorType>(
        type, place, out_dtype, ctx);
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "BuildOutputType only support DenseTensorType, SelectedRowsType, "
        "SparseCooTensorType and SparseCsrTensorType"));
  }
}

void PushBackOutputTypes(pir::IrContext* ctx,
                         pir::Operation* op_item,
                         const pir::Type& origin_type,
                         const phi::Place& out_place,
                         const phi::KernelKey& kernel_key,
                         std::vector<pir::Type>* op_output_types) {
  auto result_type = origin_type;
  if (!result_type) {
    op_output_types->push_back(result_type);
  } else if (result_type.isa<paddle::dialect::DenseTensorType>() ||
             result_type.isa<paddle::dialect::SelectedRowsType>() ||
             result_type.isa<paddle::dialect::DenseTensorArrayType>() ||
             result_type.isa<paddle::dialect::SparseCooTensorType>() ||
             result_type.isa<paddle::dialect::SparseCsrTensorType>()) {
  } else if (result_type.isa<pir::VectorType>()) {
    std::vector<pir::Type> vec_inner_types;
    auto base_types = result_type.dyn_cast<pir::VectorType>().data();
    for (auto& base_type : base_types) {
      if (base_type) {
        if (base_type.isa<paddle::dialect::DenseTensorType>() ||
            base_type.isa<paddle::dialect::SelectedRowsType>()) {
          vec_inner_types.push_back(BuildOutputType(base_type, out_place, ctx));
        } else {
          PADDLE_THROW(common::errors::Unimplemented(
              "only support dense tensor and selected rows in vector type "
              "for now"));
        }
      } else {
        pir::Type fp32_dtype = pir::Float32Type::get(ctx);
        phi::DDim dims = {};
        phi::DataLayout data_layout = phi::DataLayout::NCHW;
        phi::LegacyLoD lod = {{}};
        size_t offset = 0;
        auto dense_tensor_dtype = paddle::dialect::DenseTensorType::get(
            ctx, fp32_dtype, dims, data_layout, lod, offset);
        auto allocated_dense_tensor_dtype =
            paddle::dialect::AllocatedDenseTensorType::get(
                ctx, out_place, dense_tensor_dtype);
        vec_inner_types.push_back(allocated_dense_tensor_dtype);
      }
    }

    pir::Type t1 = pir::VectorType::get(ctx, vec_inner_types);
    op_output_types->push_back(t1);
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Result type only support DenseTensorType, SelectedRowType, "
        "SparseCooTensorType, SparseCsrTensorType and "
        "VectorType"));
  }
}
