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

#include "paddle/fluid/translator/type_translator.h"

#include "paddle/fluid/dialect/pd_type.h"
#include "paddle/fluid/dialect/pd_type_storage.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/ir/builtin_type.h"

namespace paddle {
namespace fluid {
namespace translator {

using DenseTensorType = paddle::dialect::DenseTensorType;
using DenseTensorTypeStorage = paddle::dialect::DenseTensorTypeStorage;

TypeTranslator::TypeTranslator() {
  handlers = {
      {VarType::INT64,
       [&](ir::IrContext* ctx, const VarDesc& var_desc) -> ir::Type {
         return ir::Int64Type::get(ctx);
       }},
      {VarType::FP32,
       [&](ir::IrContext* ctx, const VarDesc& var_desc) -> ir::Type {
         return ir::Float32Type::get(ctx);
       }},
      {VarType::FP64,
       [&](ir::IrContext* ctx, const VarDesc& var_desc) -> ir::Type {
         return ir::Float64Type::get(ctx);
       }},
      {VarType::LOD_TENSOR,
       [&](ir::IrContext* ctx, const VarDesc& var_desc) -> ir::Type {
         VLOG(10) << "[vartype translating]"
                  << "[" << var_desc.Name() << "]" << var_desc.GetDataType();

         ir::Type dtype =
             this->operator[](var_desc.GetDataType())(ctx, var_desc);
         DenseTensorTypeStorage::Dim dim = var_desc.GetShape();
         DenseTensorTypeStorage::DataLayout layout =
             DenseTensorTypeStorage::DataLayout::UNDEFINED;
         DenseTensorTypeStorage::LoD lod = {};
         size_t offset = 0;
         return DenseTensorType::get(ctx, dtype, dim, layout, lod, offset);
       }},
  };
}

}  // namespace translator
}  // namespace fluid
}  // namespace paddle
