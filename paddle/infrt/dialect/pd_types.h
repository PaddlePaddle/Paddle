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

// This file defines the types used in PaddlePaddle MLIR dialect.
// We borrowed much ideas from tensorflow mlir dialect (tf_types.h in
// tensorflow).

#pragma once

#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/IR/Types.h>

namespace mlir {
namespace PD {

class PaddleType : public Type {
 public:
  using Type::Type;

  static bool classof(Type type);
};

namespace detail {

template <typename Derived>
class PaddleTypeImpl : public Type::TypeBase<Derived, PaddleType, TypeStorage> {
 public:
  using Base = typename Type::TypeBase<Derived, PaddleType, TypeStorage>;
  using PDBase = PaddleTypeImpl<Derived>;
  using Base::Base;
};

}  // namespace detail

#define HANDLE_PD_TYPE(pdtype, enumerant, name)                      \
  class pdtype##Type : public detail::PaddleTypeImpl<pdtype##Type> { \
   public:                                                           \
    using PDBase::PDBase;                                            \
  };

}  // namespace PD
}  // namespace mlir
