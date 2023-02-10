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

#include "paddle/ir/type/type_support.h"

namespace ir {
/// \brief class Type
class Type {
 public:
  using ImplType = TypeStorage;

  constexpr Type() = default;

  explicit Type(const ImplType *impl) : impl_(const_cast<ImplType *>(impl)) {}

  Type(const Type &other) = default;

  Type &operator=(const Type &other) = default;

  bool operator==(Type other) const { return impl_ == other.impl_; }

  bool operator!=(Type other) const { return impl_ != other.impl_; }

  explicit operator bool() const { return impl_; }

  bool operator!() const { return impl_ == nullptr; }

  TypeId type_id() { return impl_->abstract_type().type_id(); }

  const AbstractType &abstract_type() { return impl_->abstract_type(); }

  ImplType *impl() const { return impl_; }

 protected:
  ImplType *impl_{nullptr};
};

}  // namespace ir
