
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

#pragma once

#include "paddle/pir/core/builtin_type_interfaces.h"
#include "paddle/pir/core/builtin_type_storage.h"
#include "paddle/pir/core/type.h"

namespace pir {
///
/// \brief Define built-in parameterless types.
///
/// NOTE(zhangbo9674): If you need to directly
/// cache the object of this built-in type in IrContext, please overload the get
/// method, and construct and cache the object in IrContext. For the specific
/// implementation method, please refer to Float16Type.
///
/// The built-in type object get method is as follows:
/// \code{cpp}
///   pir::IrContext *ctx = pir::IrContext::Instance();
///   Type fp32 = Float32Type::get(ctx);
/// \endcode
///

// NOTE(dev): Currently Int8 are not considered as a cached member
// in IrContextImpl because it is not widely used.

class IR_API VectorType
    : public Type::TypeBase<VectorType, Type, VectorTypeStorage> {
 public:
  using Base::Base;

  std::vector<Type> data() const;

  size_t size() const { return data().size(); }

  bool empty() const { return data().empty(); }

  Type operator[](size_t index) const { return data()[index]; }
};

class DenseTensorType : public Type::TypeBase<DenseTensorType,
                                              Type,
                                              DenseTensorTypeStorage,
                                              ShapedTypeInterface> {
 public:
  using Base::Base;

  const Type &dtype() const;

  const DenseTensorTypeStorage::Dim &dims() const;

  const DenseTensorTypeStorage::DataLayout &data_layout() const;

  const DenseTensorTypeStorage::LoD &lod() const;

  const size_t &offset() const;
};

#define DECLARE_BUILTIN_TYPE(__name)                                       \
  class IR_API __name : public Type::TypeBase<__name, Type, TypeStorage> { \
   public:                                                                 \
    using Base::Base;                                                      \
    static __name get(IrContext *context);                                 \
  };

#define FOREACH_BUILTIN_TYPE(__macro) \
  __macro(BFloat16Type);              \
  __macro(Float16Type);               \
  __macro(Float32Type);               \
  __macro(Float64Type);               \
  __macro(Int8Type);                  \
  __macro(UInt8Type);                 \
  __macro(Int16Type);                 \
  __macro(Int32Type);                 \
  __macro(Int64Type);                 \
  __macro(IndexType);                 \
  __macro(BoolType);                  \
  __macro(Complex64Type);             \
  __macro(Complex128Type);

FOREACH_BUILTIN_TYPE(DECLARE_BUILTIN_TYPE)

#undef FOREACH_BUILTIN_TYPE
#undef DECLARE_BUILTIN_TYPE

}  // namespace pir

IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::UInt8Type)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::Int8Type)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::VectorType)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::BFloat16Type)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::Float16Type)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::Float32Type)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::Float64Type)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::Int16Type)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::Int32Type)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::Int64Type)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::BoolType)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::IndexType)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::Complex64Type)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::Complex128Type)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::DenseTensorType)
