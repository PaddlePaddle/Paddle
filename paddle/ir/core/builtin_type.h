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

#include "paddle/ir/core/builtin_type_storage.h"
#include "paddle/ir/core/type.h"

namespace ir {
///
/// \brief Define built-in parameterless types. Please add the necessary
/// interface functions for built-in types through the macro
/// DECLARE_TYPE_UTILITY_FUNCTOR.
///
/// NOTE(zhangbo9674): If you need to directly
/// cache the object of this built-in type in IrContext, please overload the get
/// method, and construct and cache the object in IrContext. For the specific
/// implementation method, please refer to Float16Type.
///
/// The built-in type object get method is as follows:
/// \code{cpp}
///   ir::IrContext *ctx = ir::IrContext::Instance();
///   Type fp32 = Float32Type::get(ctx);
/// \endcode
///

// NOTE(dev): Currently Int8 are not considered as a cached member
// in IrContextImpl because it is not widely used.

class IR_API VectorType : public Type {
 public:
  using Type::Type;

  DECLARE_TYPE_UTILITY_FUNCTOR(VectorType, VectorTypeStorage);

  std::vector<Type> data() const;

  size_t size() const { return data().size(); }

  bool empty() const { return data().empty(); }

  Type operator[](size_t index) const { return data()[index]; }
};

#define DECLARE_BUILTIN_TYPE(__name)                   \
  class IR_API __name : public Type {                  \
   public:                                             \
    using Type::Type;                                  \
                                                       \
    DECLARE_TYPE_UTILITY_FUNCTOR(__name, TypeStorage); \
                                                       \
    static __name get(IrContext *context);             \
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

}  // namespace ir

IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(ir::UInt8Type)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(ir::Int8Type)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(ir::VectorType)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(ir::BFloat16Type)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(ir::Float16Type)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(ir::Float32Type)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(ir::Float64Type)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(ir::Int16Type)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(ir::Int32Type)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(ir::Int64Type)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(ir::BoolType)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(ir::IndexType)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(ir::Complex64Type)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(ir::Complex128Type)
