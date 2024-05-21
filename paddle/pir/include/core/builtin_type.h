
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

#include "paddle/pir/include/core/builtin_type_interfaces.h"
#include "paddle/pir/include/core/builtin_type_storage.h"
#include "paddle/pir/include/core/type.h"

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
  static std::string name() { return "t_vec"; }

  size_t size() const { return data().size(); }

  bool empty() const { return data().empty(); }

  Type operator[](size_t index) const { return data()[index]; }
};

class IR_API DenseTensorType : public Type::TypeBase<DenseTensorType,
                                                     Type,
                                                     DenseTensorTypeStorage,
                                                     ShapedTypeInterface> {
 public:
  using Base::Base;
  using Dim = DenseTensorTypeStorage::Dim;
  using LoD = DenseTensorTypeStorage::LoD;

  Type dtype() const;
  const Dim &dims() const;
  DataLayout data_layout() const;
  const LoD &lod() const;
  size_t offset() const;
  static std::string name() { return "t_dtensor"; }
  ///
  /// \brief Implementation of 'classof' that compares the type id of
  /// the provided value with the concrete type id.
  ///
  static bool classof(Type type);

  static DenseTensorType dyn_cast_impl(Type type);

  static DenseTensorType get(IrContext *ctx,
                             Type dtype,
                             const Dim &dims,
                             DataLayout layout = DataLayout::kNCHW,
                             const LoD &lod = {},
                             size_t offset = 0u) {
    return Base::get(ctx, dtype, dims, layout, lod, offset);
  }
};

#define DECLARE_BUILTIN_TYPE(__name, s_name)                               \
  class IR_API __name : public Type::TypeBase<__name, Type, TypeStorage> { \
   public:                                                                 \
    using Base::Base;                                                      \
    static __name get(IrContext *context);                                 \
    static std::string name() { return s_name; }                           \
  };

#define FOREACH_BUILTIN_TYPE(__macro) \
  __macro(BFloat16Type, "t_bf16");    \
  __macro(Float16Type, "t_f16");      \
  __macro(Float32Type, "t_f32");      \
  __macro(Float64Type, "t_f64");      \
  __macro(Int8Type, "t_i8");          \
  __macro(UInt8Type, "t_ui8");        \
  __macro(Int16Type, "t_i16");        \
  __macro(Int32Type, "t_i32");        \
  __macro(Int64Type, "t_i64");        \
  __macro(IndexType, "t_index");      \
  __macro(BoolType, "t_bool");        \
  __macro(Complex64Type, "t_c64");    \
  __macro(Complex128Type, "t_c128");
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
