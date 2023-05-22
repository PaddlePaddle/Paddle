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

#include "paddle/ir/builtin_type_storage.h"
#include "paddle/ir/type.h"

namespace ir {
///
/// \brief This macro is used to get a list of all built-in types in this file.
/// The built-in Dialect will use this macro to quickly register all built-in
/// types.
///
#define GET_BUILT_IN_TYPE_LIST                                              \
  BFloat16Type, Float16Type, Float32Type, Float64Type, Int8Type, Int16Type, \
      Int32Type, Int64Type, BoolType, VectorType

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
class BFloat16Type : public Type {
 public:
  using Type::Type;

  DECLARE_TYPE_UTILITY_FUNCTOR(BFloat16Type, TypeStorage);
};

class Float16Type : public Type {
 public:
  using Type::Type;

  DECLARE_TYPE_UTILITY_FUNCTOR(Float16Type, TypeStorage);

  static Float16Type get(IrContext *context);
};

class Float32Type : public Type {
 public:
  using Type::Type;

  DECLARE_TYPE_UTILITY_FUNCTOR(Float32Type, TypeStorage);

  static Float32Type get(IrContext *context);
};

class Float64Type : public Type {
 public:
  using Type::Type;

  DECLARE_TYPE_UTILITY_FUNCTOR(Float64Type, TypeStorage);

  static Float64Type get(IrContext *context);
};

class Int8Type : public Type {
 public:
  using Type::Type;

  DECLARE_TYPE_UTILITY_FUNCTOR(Int8Type, TypeStorage);
};

class Int16Type : public Type {
 public:
  using Type::Type;

  DECLARE_TYPE_UTILITY_FUNCTOR(Int16Type, TypeStorage);

  static Int16Type get(IrContext *context);
};

class Int32Type : public Type {
 public:
  using Type::Type;

  DECLARE_TYPE_UTILITY_FUNCTOR(Int32Type, TypeStorage);

  static Int32Type get(IrContext *context);
};

class Int64Type : public Type {
 public:
  using Type::Type;

  DECLARE_TYPE_UTILITY_FUNCTOR(Int64Type, TypeStorage);

  static Int64Type get(IrContext *context);
};

class BoolType : public Type {
 public:
  using Type::Type;

  DECLARE_TYPE_UTILITY_FUNCTOR(BoolType, TypeStorage);

  static BoolType get(IrContext *context);
};

class VectorType : public Type {
 public:
  using Type::Type;

  DECLARE_TYPE_UTILITY_FUNCTOR(VectorType, VectorTypeStorage);

  std::vector<Type> data() const;

  size_t size() const { return data().size(); }

  bool empty() const { return data().empty(); }

  Type operator[](size_t index) const { return data()[index]; }
};

}  // namespace ir
